import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
import os

st.set_page_config(layout="wide", page_title="Time Series Visualizer (Styled)")

# ---------------------------------------------------------
# 1. 데이터 로드 (기존 동일)
# ---------------------------------------------------------
FILE_PATH = {
    'ETTh1': './dataset/ETTh1.csv',
    'ETTh2': './dataset/ETTh2.csv',
    'ETTm1': './dataset/ETTm1.csv',
    'ETTm2': './dataset/ETTm2.csv',
    'ECL': './dataset/electricity.csv',
    'Weather': './dataset/weather.csv',
    'Traffic': './dataset/traffic.csv',
    'ILI': './dataset/national_illness.csv',
    'Exchange': './dataset/exchange_rate.csv',
    'PEMS03': './dataset/PEMS03.npz',
    'PEMS04': './dataset/PEMS04.npz',
    'PEMS07': './dataset/PEMS07.npz',
    'PEMS08': './dataset/PEMS08.npz',
}

@st.cache_data
def load_data(dataset_name: str):
    if dataset_name not in FILE_PATH: return None, None, None
    real_path = FILE_PATH[dataset_name]
    
    if not os.path.exists(real_path):
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="H")
        data = np.random.randn(1000, 5).cumsum(axis=0)
        return dates, data, [f"Channel {i}" for i in range(5)]

    if real_path.endswith('.npz'):
        with np.load(real_path) as data:
            raw_data = data['data']
            data_values = raw_data[:, :, 0] if raw_data.ndim == 3 else raw_data
            dates = pd.date_range(start="2020-01-01", periods=data_values.shape[0], freq="5min")
            columns = [f"Sensor {i}" for i in range(data_values.shape[1])]
            return dates, data_values, columns
    elif real_path.endswith('.csv'):
        df = pd.read_csv(real_path)
        dates = pd.to_datetime(df[df.columns[0]])
        data_values = df.iloc[:, 1:].values
        columns = df.columns[1:].tolist()
        return dates, data_values, columns
    return None, None, None

# ---------------------------------------------------------
# 2. UI 및 시각화 로직
# ---------------------------------------------------------
st.sidebar.title("🛠️ 데이터 컨트롤러")
dataset_name = st.sidebar.selectbox("📂 데이터셋 선택", list(FILE_PATH.keys()))
dates, data, col_names = load_data(dataset_name)

if data is not None:
    st.sidebar.subheader("📊 채널 선택")
    default_vals = [col_names[0], col_names[1]] if len(col_names) > 1 else col_names[:1]
    
    raw_selected = st.sidebar.multiselect("채널 선택", col_names, default=default_vals)
    selected_channels = sorted(raw_selected, key=lambda x: col_names.index(x))

    if not selected_channels:
        st.stop()

# -----------------------------------------------------
    # [최적화] 데이터 다운샘플링
    # -----------------------------------------------------
    MAX_POINTS = 5000
    total_len = len(dates)
    
    if total_len > MAX_POINTS:
        step = total_len // MAX_POINTS
        # Pandas 객체 슬라이싱
        display_dates = dates[::step]
        display_data = data[::step]
        st.sidebar.caption(f"🚀 성능 최적화: {total_len:,}개 → {len(display_dates):,}개로 샘플링")
    else:
        step = 1
        display_dates = dates
        display_data = data
        
    # [🔥 핵심 수정 사항] 
    # Pandas Series/Index 상태에서는 [] 접근이 라벨 검색일 수 있으므로,
    # 안전하게 Numpy 배열로 변환하여 순서(Position) 기반 인덱싱이 되도록 강제합니다.
    if isinstance(display_dates, (pd.Series, pd.Index)):
        display_dates = display_dates.values  # 혹은 .to_numpy()

    # 날짜 문자열 변환
    dates_str = pd.Series(display_dates).dt.strftime('%Y-%m-%d %H:%M').tolist()
    
    # 현재 화면에 표시되는 실제 데이터 길이 (샘플링 후)
    current_display_len = len(display_dates)

    # -----------------------------------------------------
    # [🎨 상단 정보 패널: 슬라이더 상태 표시]
    # -----------------------------------------------------
    # 초기값 설정 (0% ~ 100%)
    start_percent = 0
    end_percent = 100

    # 나중에 차트 이벤트에서 값을 받아오면 덮어씌움
    if "zoom_state" not in st.session_state:
        st.session_state["zoom_state"] = {"start": 0, "end": 100}

    # 상단에 정보를 띄울 공간 확보
    info_container = st.container()

    # -----------------------------------------------------
    # [ECharts 옵션 생성]
    # -----------------------------------------------------
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Legend 생성
    st.sidebar.markdown("---")
    legend_html = ""
    for i, channel in enumerate(selected_channels):
        color = COLORS[i % len(COLORS)]
        legend_html += f"""
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 15px; height: 15px; background-color: {color}; margin-right: 10px; border-radius: 3px;"></div>
            <span style="font-weight: bold; color: #333;">{channel}</span>
        </div>
        """
    st.sidebar.markdown(legend_html, unsafe_allow_html=True)

    # 레이아웃 계산
    num_channels = len(selected_channels)
    CHART_HEIGHT = 220
    SLIDER_HEIGHT = 40
    GAP_BETWEEN = 10
    MARGIN_BOTTOM = 40
    UNIT_HEIGHT = CHART_HEIGHT + GAP_BETWEEN + SLIDER_HEIGHT + MARGIN_BOTTOM
    total_height = num_channels * UNIT_HEIGHT

    option = {
        "animation": False,
        "hoverLayerThreshold": 3000,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "grid": [],
        "xAxis": [],
        "yAxis": [],
        "series": [],
        # 줌 이벤트는 시리즈별로 적용되지만, 슬라이더는 하나로 통합
        "dataZoom": [{"type": "inside", "xAxisIndex": list(range(num_channels)), "zoomOnMouseWheel": True}]
    }

    for i, channel_name in enumerate(selected_channels):
        col_idx = col_names.index(channel_name)
        series_data = display_data[:, col_idx].tolist()
        current_color = COLORS[i % len(COLORS)]

        current_top = i * UNIT_HEIGHT
        slider_top = current_top + CHART_HEIGHT + GAP_BETWEEN

        option["grid"].append({
            "left": "130px", "right": "1%", "top": current_top, "height": CHART_HEIGHT, "containLabel": False
        })

        option["xAxis"].append({
            "type": "category",
            "boundaryGap": False,
            "data": dates_str,
            "gridIndex": i,
            "axisLabel": {"show": False},
            "axisTick": {"show": False}
        })

        option["yAxis"].append({
            "type": "value",
            "gridIndex": i,
            "name": channel_name,
            "nameLocation": "middle",
            "nameRotate": 0,
            "nameGap": 80,
            "nameTextStyle": {"color": "#000000", "fontWeight": "bold", "fontSize": 14, "align": "right"},
            "splitLine": {"show": True, "lineStyle": {"type": "dashed", "opacity": 0.5}}
        })

        option["series"].append({
            "name": channel_name,
            "type": "line",
            "xAxisIndex": i,
            "yAxisIndex": i,
            "data": series_data,
            "showSymbol": False,
            "lineStyle": {"width": 1.5, "color": current_color},
            "itemStyle": {"color": current_color},
            "sampling": "lttb",
            "silent": True
        })

        # DataZoom Slider 설정
        option["dataZoom"].append({
            "type": "slider",
            "xAxisIndex": list(range(num_channels)), 
            "top": slider_top,
            "height": SLIDER_HEIGHT,
            
            # 이전에 저장된 줌 상태 유지 (리런 시 초기화 방지)
            "start": st.session_state["zoom_state"]["start"],
            "end": st.session_state["zoom_state"]["end"],
            
            "showDataShadow": True, 
            "labelFormatter": None, # 자동 날짜 표시
            "dataBackground": {
                "lineStyle": {"color": current_color, "opacity": 0.6},
                "areaStyle": {"color": current_color, "opacity": 0.2}
            },
            "borderColor": "transparent",
            "backgroundColor": "#f5f5f5",
            "handleSize": "100%"
        })

    # -----------------------------------------------------
    # [이벤트 핸들링 및 차트 렌더링]
    # -----------------------------------------------------
    # dataZoom 이벤트를 캡처하여 start, end 값을 리턴받습니다.
    # params.batch[0]에 start(%), end(%) 정보가 들어있습니다.
    events = {
        "dataZoom": """
        function(params) {
            // 1. 마우스 휠 줌 (batch 배열에 담겨옴)
            if (params.batch && params.batch.length > 0) {
                return {
                    start: params.batch[0].start,
                    end: params.batch[0].end
                };
            }
            // 2. 슬라이더 드래그 (root 레벨에 start/end 존재)
            if (params.start !== undefined && params.end !== undefined) {
                return {
                    start: params.start,
                    end: params.end
                };
            }
            // 3. 예외 케이스 방지 (null 리턴 방지)
            return {start: 0, end: 100, error: "unknown event format"};
        }
        """
    }

    # 차트 그리기
    chart_event = st_echarts(
        options=option, 
        height=f"{total_height}px",
        events=events,
        key=f"chart_{dataset_name}_{len(selected_channels)}"
    )

    # -----------------------------------------------------
    # [상단 정보 업데이트 로직]
    # -----------------------------------------------------
    # 1. 차트 이벤트 수신 및 세션 업데이트
    if chart_event and isinstance(chart_event, dict):
        new_start = chart_event.get("start")
        new_end = chart_event.get("end")
        
        # 값이 정상적으로 숫자형태로 왔을 때만 업데이트
        if new_start is not None and new_end is not None:
            st.session_state["zoom_state"]["start"] = new_start
            st.session_state["zoom_state"]["end"] = new_end
    
    # -----------------------------------------------------
    # [디버깅용: 만약 여전히 안 된다면 아래 주석을 풀어보세요]
    # st.write("Debug Event:", chart_event) 
    # -----------------------------------------------------

    curr_start_pct = st.session_state["zoom_state"]["start"]
    curr_end_pct = st.session_state["zoom_state"]["end"]

    # 2. 샘플링된 데이터 기준 인덱스 계산
    sampled_start_idx = int(current_display_len * (curr_start_pct / 100))
    sampled_end_idx = int(current_display_len * (curr_end_pct / 100))
    
    # 인덱스 범위 보호
    sampled_start_idx = max(0, min(sampled_start_idx, current_display_len - 1))
    sampled_end_idx = max(0, min(sampled_end_idx, current_display_len - 1))

    # 3. 원본 데이터 기준 인덱스 및 크기 복원
    real_start_idx = sampled_start_idx * step
    real_end_idx = min(sampled_end_idx * step, total_len)
    real_window_len = real_end_idx - real_start_idx
    
    # 4. 날짜 정보 가져오기
    start_date_display = display_dates[sampled_start_idx]
    end_date_display = display_dates[sampled_end_idx]
    
    start_ts = pd.to_datetime(start_date_display)
    end_ts = pd.to_datetime(end_date_display)

    # -----------------------------------------------------
    # [지표 표시]
    # -----------------------------------------------------
    with info_container:
        st.header(f"📈 {dataset_name}")
        c1, c2, c3, c4, c5 = st.columns(5)
        
        c1.metric("Total Data Length", f"{total_len:,}") 
        c2.metric("Start Index", f"{real_start_idx:,}")
        
        # [확인] 이제 슬라이더를 놓으면 이 값이 바뀔 것입니다.
        c3.metric("Current Window Size", f"{real_window_len:,}") 
        
        c4.metric("Start Date", start_ts.strftime('%Y-%m-%d %H:%M'))
        c5.metric("End Date", end_ts.strftime('%Y-%m-%d %H:%M'))
        
        st.markdown("---")

else:
    st.error("데이터 로드 실패")
