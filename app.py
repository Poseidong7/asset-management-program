import streamlit as st
import os
import datetime
import time 
import json # 대화 기록 저장용 라이브러리
import pandas as pd #데이터 분석 > 표 만들기
import yfinance as yf #[신규] 주식 정보를 가져오는 라이브러리
import re # 정규표현식 모듈 

# --- [필수 라이브러리] ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- [설정] ---
os.environ["GOOGLE_API_KEY"] = "" 

# 데이터 저장소 경로
DATA_PATH = "./Data_Vault"
DB_PATH = "./chroma_db"
CHAT_LOG_FILE = "./chat_history.json" # [추가] 대화 기록 저장할 파일

# --- [UI 꾸미기] ---
st.set_page_config(page_title="D.O.N.G.V.I.S.", page_icon="🐎", layout="wide")
st.title("🐎 D.O.N.G.V.I.S. : 나만의 AI 비서")

# --- [★추가] 주식 이름 사전 (주요 50개 + 사용자 커스텀) ---
STOCK_MAP = {
    #[사용자 보유 종목 추가 공간]
    #형식 : "종목명" : "코드번호.KS or .KQ"
    "내주식1": "000000.KS", # 예시

    # [한국 주식]
    "삼성전자": "005930.KS", "삼전": "005930.KS",
    "SK하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "현대차": "005380.KS", "기아": "000270.KS",
    "NAVER": "035420.KS", "카카오": "035720.KS", 
    "LG에너지솔루션": "373220.KS", "POSCO홀딩스": "005490.KS",
    "삼성바이오로직스": "207940.KS", "셀트리온": "068270.KS",
    "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ",
    "알테오젠": "196170.KQ", "HLB": "028300.KQ",
    "두산에너빌리티": "034020.KS", "한화에어로스페이스": "012450.KS",

    # [미국 주식]
    "애플": "AAPL", "마이크로소프트": "MSFT", "엔비디아": "NVDA", 
    "구글": "GOOGL", "아마존": "AMZN", "테슬라": "TSLA", "메타": "META",
    "TSMC": "TSM", "넷플릭스": "NFLX", "코카콜라": "KO", "스타벅스": "SBUX",
    "리얼티인컴": "O", "SCHD": "SCHD", "SPY": "SPY", "QQQ": "QQQ",
    "TQQQ": "TQQQ", "SOXL": "SOXL", "아이온큐": "IONQ", "팔란티어": "PLTR",

    # [가상화폐]
    "비트코인": "BTC-USD", "비트": "BTC-USD",
    "이더리움": "ETH-USD", "이더": "ETH-USD",
    "리플": "XRP-USD", "솔라나": "SOL-USD", "도지코인": "DOGE-USD"
}

# --- [함수 모음] ---

# 윈도우 금지 문자를 모두 언더바로 교체
def clean_filename(filename):
    cleaned = re.sub(r'[\\/:*?"<>|]', '_', filename)
    return cleaned

def load_chat_history():
    """앱 켜질 때 지난 대화 불러오기"""
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_chat_history(messages):
    """대화 한마디 할 때마다 파일에 저장하기"""
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# --- [메모리(기억) & DB 로드 함수] ---
@st.cache_resource
def load_db():
    # 폴더가 없으면 아예 로드하지 않음 (에러 방지)
    if not os.path.exists(DB_PATH):
        return None
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
        return vectordb
    except Exception as e:
        # DB가 깨졌으면 None 반환
        return None

# --- [2. 파일 쓰기 함수] ---
def save_to_file(category, content):
    safe_category = clean_filename(category)

    target_folder = os.path.join(DATA_PATH, safe_category)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    file_path = os.path.join(target_folder, "자동기록.txt")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{now}] {content}")

    return f"✅ 기록 완료! ({category}/자동기록.txt)"

# --- [문서 포맷팅 함수] ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- [타자 효과 함수] ---
def stream_text(text):
    for chunk in text.split(" "): 
        yield chunk + " "
        time.sleep(0.05)

# [★신규] 실시간 환율 가져오기
def get_exchange_rate():
    try:
        ticker = yf.Ticker("KRW=X") #원달러 환율 코드
        data = ticker.history(period="1d")
        if not data.empty:
            rate = data['Close'].iloc[-1]
            return rate
        return 1450.0 #조회 실패시 기본값
    except:
        return 1450.0

# [변경] 이름 -> 코드 = 가격 출력
def get_stock_price(ticker_name):
    ticker_code = ticker_name 

    # 사전에 있는지 확인 (공백 제거 후 검색)
    clean_name = ticker_name.strip()
    if clean_name in STOCK_MAP:
        ticker_code = STOCK_MAP[clean_name]
    # 숫자만 입력했다면 한국 주식으로 가정
    elif ticker_name.isdigit():
        ticker_code += ".KS"

    try:
        stock = yf.Ticker(ticker_code)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1], ticker_code
        else:
            return None, ticker_code
    except:
        return None, ticker_code
    

# --- 자산 데이터 분석 함수 ---
def analyze_assets_with_ai():
    """모든 텍스트 파일을 읽어서 AI에게 분석 요청"""
    # 1. 파일이 있는지 확인
    if not os.path.exists(DATA_PATH):
        return None, "데이터가 없습니다."
    
    # 2. 모든 텍스트 파일 내용을 하나로 합치기
    all_text = ""
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    all_text += f"\n--- [{file}] ---\n{f.read()}"

    if not all_text.strip():
        return None, "기록된 내용이 없습니다."
    
    # 3. AI에게 "JSON 형식으로 분석해줘"라고 명령
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # [변경] 평단가(avg_price) 추출 및 부채/지출 기록 인식 강화
    prompt = f"""
    아래는 사용자의 재무/자산 기록들이야. 이걸 분석해서 JSON 형식으로 요약해줘.
    [기록 내용] {all_text}
    [요청 사항]
    1. '유동성자산', '투자자산', '부동자산', '부채' 4가지로 분류해 합산해.
    2. ★중요: [부채] 폴더에 있는 내용이나 '썼다', '지출', '카드' 관련 기록은 모두 '부채'로 합산해.
    3. '총자산' = (유동성+투자+부동), '순자산' = (총자산-부채).
    4. ★매우중요: 사용자가 보유한 '주식'이나 '코인' 종목의 [이름, 수량, 구매가격(평단가)]를 추출해줘.
       - '매수', '샀다'는 수량 추가, '매도', '팔았다'는 수량 차감.
       - 구매가격을 모르면 0으로 해.
    5. 반드시 아래 JSON 형식으로만 출력해.
    {{
        "total_asset": 0, "net_asset": 0, "debt": 0,
        "details": {{ "유동성자산": 0, "투자자산": 0, "부동자산": 0, "부채": 0 }},
        "holdings": [
            {{"name": "삼성전자", "qty": 12, "avg_price": 70000}}, 
            {{"name": "비트코인", "qty": 0.1, "avg_price": 80000}}
        ],
        "advice": "조언"
    }}
    """

    try:
        response = llm.invoke(prompt).content
        cleaned_response = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_response) #문자를 진짜 데이터로 변환
        return data, "성공"
    except Exception as e:
        return None, f"분석 중 오류 발생: {e}"
    

# ==========================================
#         [1. 사이드바 - 설정 구역]
# ==========================================
with st.sidebar:
        st.header("🧠 두뇌 관리")
        if st.button("지식 업데이트 (뇌 세척)"):
            with st.spinner("새로운 지식을 흡수하는 중..."):
                #1. 데이터 폴더 확인
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)

                #2. 파일 읽어오기
                loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
                documents = loader.load()

                if documents:
                    #3. 텍스트 쪼개기
                    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    texts = text_splitter.split_documents(documents)
                    
                    #4. 임베딩 모델 준비
                    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
                    
                    #[수정] 삭제 -> 덮어쓰기 시도
                    # DB가 있으면 연결, 없으면 생성
                    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
                    
                    try:
                        #기존 데이터가 존재하면 지우고 새로 넣기
                        # get()으로 모든 ID를 가져와 삭제하는 방식
                        existing_ids = vectordb.get()['ids']
                        if existing_ids:
                            vectordb.delete(ids=existing_ids)

                        #새로운 데이터 넣기
                        vectordb.add_documents(texts)

                        st.success("업데이트 완료! (폴더 안 지워도 됩니다 👍)")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        #윈도우 파일 잠금 때문에 실패 경우 대비
                        st.error(f"업데이트 중 오류가 발생했습니다: {e}")
                        st.warning("혹시 해결이 안 되면 터미널을 껐다가 다시 켜주세요. (윈도우 파일 잠금 문제)")
                else:
                    st.warning("Data_Vault 폴더가 비어있습니다. 텍스트 파일을 넣어주세요.")

        st.divider() #구분선
    
        #[추가] 비서 성격 선택 -> 클릭 방식으로 변경
        st.header("🎭 페르소나 설정")
        persona_mode = st.radio(
            "비서의 성격을 선택하세요:",
            ("차분한 비서 (기본)", "스파르타 조교 (팩트폭행)", "다정한 엄마 (걱정인형)"),
            index=0 # 기본값: 첫 번째
        )

        st.divider() #구분선

        #[대화 초기화 버튼 추가]
        st.header("💬 대화 관리")
        if st.button("대화 내용 지우기 (초기화)"):
            st.session_state.messages = [] #화면에서 지우기
            if os.path.exists(CHAT_LOG_FILE):
                os.remove(CHAT_LOG_FILE)
            st.rerun()

# ==========================================
#          [2. 메인 화면 - 탭 구역]
# ==========================================
tab1, tab2 = st.tabs(["💬 대화하기", "📊 자산 대시보드"])
with tab1:
    #1. 대화 내용 표시될 그릇 만들기
    chat_container = st.container()

    # 2. 지난 대화 불러오기
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()

    # 3. 대화 화면에 출력
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 4. 입력창 및 답변 로직
    if prompt := st.chat_input("입력해주세요!"):

        # 사용자 메시지 표시 및 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_history(st.session_state.messages)
        

        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # AI 답변 생성
        with st.chat_message("assistant"):
            vectordb = load_db()
            final_response = "" #변수 초기화 (에러 방지)

            if vectordb:
                try:
                    with st.spinner("열심히 생각 중... 🧠"):
                        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
                        
                        current_date = datetime.datetime.now().strftime("%Y년 %m월 %d일")

                        # [성격 설정]
                        style_guide = ""
                        if "스파르타" in persona_mode:
                            style_guide = """
                            너는 '지옥에서 온 재무 트레이너'야. 반말을 써.
                            돈을 쓰면 "정신 차려라", "그게 필요하냐"라고 아주 따끔하게 혼내줘.
                            """
                        elif "엄마" in persona_mode:
                            style_guide = """
                            너는 '걱정 많은 엄마'야. '우리 아들/딸'이라고 부르고,
                            돈을 쓰면 "아이고 아껴야지"라고 따뜻하게 잔소리해줘.
                            """
                        else:
                            style_guide = "너는 정중하고 유능한 개인 비서야. 존댓말을 써."

                        # [★핵심 수정] 저장 규칙을 AI에게 명확히 교육!
                        template = f"""
                        {style_guide}
                        
                        [현재 날짜]: {current_date}
                        
                        [★자동 기록 규칙]
                        1. 돈을 썼다, 결제했다, 밥 먹었다 -> [SAVE:부채] 에 기록해. (카드값/지출로 인식)
                        2. 주식/코인을 샀다, 매수했다 -> [SAVE:주식기록] 에 기록해.
                        3. 월급 받았다, 입금됐다 -> [SAVE:유동성자산] 에 기록해.
                        4. 그 외 일반적인 내용 -> [SAVE:메모] 에 기록해.
                        
                        [특별 지시]
                        답변할 때 "[SAVE:폴더명] 내용" 형식을 맨 마지막에 꼭 붙여줘.
                        모든건 팩트 그리고 철저한 분석을 통해서 말해줘
                        
                        [참고 문서]
                        {{context}}
                        
                        질문: {{question}}
                        답변:
                        """
                        
                        custom_prompt = PromptTemplate.from_template(template)

                        rag_chain = (
                            {"context": retriever | format_docs, "question": RunnablePassthrough()}
                            | custom_prompt
                            | llm
                            | StrOutputParser()
                        )

                        response = rag_chain.invoke(prompt)
                        final_response = response
                        
                        # 저장 로직
                        if "[SAVE:" in final_response:
                            try:
                                #1. 정규표현식을 사용해 패턴을 정확히 찾음
                                found_match = re.search(r"\[SAVE:(.*?)\]", final_response)

                                if found_match:
                                    #2. 괄호 안의 내용만 뽑기
                                    raw_header = found_match.group(1).strip()

                                    #3. 괄호가 끝난 뒤의 나머지 부분 추출
                                    content = final_response[found_match.end():].strip()

                                    #4. 저장 함수 실행
                                    result_msg = save_to_file(raw_header, content)

                                    #5. 결과 메세지로 덮어쓰기
                                    final_response = f"{result_msg}\n\n내용: {content}"
                                else:
                                    final_response += "\n(⚠️ 저장 실패: 형식이 올바르지 않습니다.)"

                            except Exception as e:
                                 final_response += f"\n(❌ 저장 시스템 오류: {e})"
                                

                    st.write_stream(stream_text(final_response))
                
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
                    final_response = "오류 발생 (지식 업데이트 필요)"
            else:
                final_response = "지식 DB가 없습니다. 왼쪽의 [지식 업데이트] 버튼을 먼저 눌러주세요!"
                st.write(final_response)

        st.session_state.messages.append({"role": "assistant", "content": final_response})
        save_chat_history(st.session_state.messages) # [저장]

# --- [Tab 2] 자산 대시보드 ---
with tab2:
    st.header("📊 내 자산 현황판")
    st.caption("AI가 기록된 메모를 읽고 분석한 결과입니다.")

    # [★핵심 기능] 분석 결과를 세션(Session)에 저장하기 (화면 고정용)
    if 'dashboard_data' not in st.session_state:
        st.session_state['dashboard_data'] = None
    if 'dashboard_rate' not in st.session_state:
        st.session_state['dashboard_rate'] = 1450.0

    # 1. 자산 분석 버튼 (누르면 세션에 데이터 저장!)
    st.subheader("1. 전체 자산 분석 & 수익률 확인")
    if st.button("🔄 자산 & 수익률 분석 실행"):
        with st.spinner("AI가 자산을 분석하고 인터넷에서 현재가를 가져옵니다... 🧮"):
            
            # 환율 조회 및 저장
            rate = get_exchange_rate()
            st.session_state['dashboard_rate'] = rate
            st.toast(f"💵 현재 적용 환율: {rate:,.2f} 원")

            # 데이터 분석 및 저장
            data, msg = analyze_assets_with_ai()
            if data:
                st.session_state['dashboard_data'] = data
            else:
                st.error(msg)
    
    # [★화면 그리기] 저장된 데이터가 있으면 무조건 그린다 (버튼 안 눌러도!)
    if st.session_state['dashboard_data']:
        data = st.session_state['dashboard_data']
        usd_krw_rate = st.session_state['dashboard_rate']

        # [1] 전체 요약
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 총 자산", f"{data['total_asset']:,}원")
        col2.metric("📉 부채 (빚)", f"{data['debt']:,}원", delta_color="inverse")
        col3.metric("💎 순자산", f"{data['net_asset']:,}원")
        st.divider()

        # [2] 자산 구성 그래프
        df = pd.DataFrame(list(data['details'].items()), columns=["카테고리", "금액"])
        df.set_index("카테고리", inplace=True)
        st.bar_chart(df)
        st.divider()

        # [3] ★보유 주식 실시간 평가 (수익률 추가!)
        st.subheader("📈 내 보유 주식 수익률 (P&L)")
        holdings = data.get("holdings", [])

        if holdings:
            stock_list = []
            total_stock_value = 0

            for item in holdings:
                name = item['name']
                qty = item['qty']
                avg_price = item.get('avg_price', 0) #AI가 찾은 평단가

                #이름 찾아서 현재가
                current_price, ticker = get_stock_price(name)

                if current_price:
                    # [변경] 환율 및 수익률 로직 적용
                    display_current_price = current_price
                    display_avg_price = avg_price
                    currency = "원"
                    
                    #미국 주식이면 원화로 환산
                    is_foreign = "USD" in ticker or ticker in ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "AMZN", "SBUX", "KO", "O", "SCHD", "SPY", "QQQ"]
                    
                    if is_foreign:
                        display_current_price = current_price * usd_krw_rate 
                        # 평단가가 달러로 적혀있다면(대충 5000원 이하) 환율 곱해주기
                        if avg_price < 5000 and avg_price > 0:
                            display_avg_price = avg_price * usd_krw_rate
                        currency = "원(환산)"
                        
                    val = display_current_price * qty
                    total_stock_value += val
                    
                    # [핵심] 수익률 계산
                    profit_rate = 0
                    profit_val = 0
                    if display_avg_price > 0:
                        profit_rate = ((display_current_price - display_avg_price) / display_avg_price) * 100
                        profit_val = val - (display_avg_price * qty)

                    stock_list.append({
                        "종목명": name,
                        "수량": qty,
                        "평단가": f"{display_avg_price:,.0f}",
                        "현재가": f"{display_current_price:,.0f} {currency}",
                        "평가액": f"{val:,.0f}",
                        "수익률": f"{profit_rate:+.2f}%", 
                        "손익": f"{profit_val:+,.0f}"   
                    })
                else:
                    stock_list.append({
                        "종목명": name,
                        "수량": qty,
                        "평단가": f"{avg_price}",
                        "현재가": "조회불가",
                        "평가액": "-", "수익률": "-", "손익": "-"
                    })
            
            # 데이터프레임으로 출력
            st.dataframe(pd.DataFrame(stock_list), use_container_width=True)
            st.info(f"💰 보유 주식 총 평가액(추정): {total_stock_value:,.0f} 원")

        else:
            st.caption("기록된 주식이 없습니다. 본인 보유 자산을 기록하세요")
        
        st.success(f"AI 조언: {data['advice']}")

    st.divider()

    # 개별 종목 검색
    st.subheader("🔍 개별 종목 시세 조회")
    col_input, col_btn = st.columns([3, 1])

    with col_input:
        ticker_input = st.text_input("종목명 입력", placeholder="예: 삼전, 슈드, 비트코인")

    with col_btn:
        st.write("")
        st.write("")
        search_btn = st.button("가격 확인")

    if search_btn and ticker_input:
        with st.spinner("가격을 알아보는 중..."):
            #환율 조회 (저장된 값 사용 또는 재조회)
            usd_krw_rate = st.session_state.get('dashboard_rate', 1450.0)
            price, ticker_code = get_stock_price(ticker_input)

            if price:
                #미국 주식이면 원화 환산 가격도 같이 출력
                if "USD" in ticker_code or ticker_code.isalpha():
                    krw_price = price * usd_krw_rate
                    st.success(f"🔎 [{ticker_input}] 현재가")
                    st.metric(label="가격", value=f"{price:,.2f} USD", delta=f"약 {krw_price:,.0f} 원")
                    st.caption(f"적용 환율: {usd_krw_rate:,.2f} 원")
                else:
                    st.success(f"🔎 [{ticker_input}] 현재가")
                    st.metric(label="가격", value=f"{price:,.0f} 원")
            else:
                st.error("종목을 찾을 수 없습니다.")