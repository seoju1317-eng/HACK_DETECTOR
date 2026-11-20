import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="치트 의심 영상 분석기", page_icon="🕵️", layout="wide")

ANALYZE_W = 256  # 분석용 축소 해상도(가로)


# ---------------------------------------------------------------- 분석 함수
def analyze_motion(path, stride, max_sec, progress):
    """프레임 간 전역 광학 흐름(카메라/조준 이동 속도)을 계산."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total, int(max_sec * fps)) if max_sec else total
    dt = stride / fps
    vx, vy, idxs = [], [], []
    hcorr, bright, bdiff, local, ncc, mad = [], [], [], [], [], []
    prev, prev_hist, i = None, None, 0
    while i < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (ANALYZE_W, max(1, int(h * ANALYZE_W / w))))
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([g], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if prev is not None:
                f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 5, 31, 5, 7, 1.5, 0)
                # 화면폭/초 단위로 정규화 (중앙값 = 카메라 전체 이동에 강건)
                dxp, dyp = float(np.median(f[..., 0])), float(np.median(f[..., 1]))
                vx.append(dxp / ANALYZE_W / dt)
                vy.append(dyp / ANALYZE_W / dt)
                idxs.append(i)
                hc = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                hcorr.append(0.0 if np.isnan(hc) else float(hc))
                bright.append(float(g.mean()))
                mad.append(float(cv2.absdiff(g, prev).mean()))
                bdiff.append(abs(float(g.mean()) - float(prev.mean())))
                M = np.float32([[1, 0, dxp], [0, 1, dyp]])
                warped = cv2.warpAffine(prev, M, (g.shape[1], g.shape[0]), borderMode=cv2.BORDER_REPLICATE)
                local.append(float((cv2.absdiff(g, warped) > 35).mean()))
                nv = float(cv2.matchTemplate(g, warped, cv2.TM_CCOEFF_NORMED)[0, 0])
                ncc.append(0.0 if np.isnan(nv) else nv)
            prev, prev_hist = g, hist
        i += 1
        if i % 30 == 0:
            progress.progress(min(i / max(limit, 1), 1.0))
    cap.release()
    extra = dict(hcorr=np.array(hcorr), bright=np.array(bright),
                 bdiff=np.array(bdiff), local=np.array(local), ncc=np.array(ncc),
                 mad=np.array(mad))
    return fps, dt, np.array(vx), np.array(vy), np.array(idxs), extra


def detect_snaps(vx, vy, dt, thr):
    """급가속 → 급정지 형태의 '스냅' 이벤트를 탐지하고 특성을 계산."""
    s = np.hypot(vx, vy)
    peaks = []
    for i in range(1, len(s) - 1):
        if s[i] >= thr and s[i] >= s[i - 1] and s[i] > s[i + 1]:
            if peaks and (i - peaks[-1]) * dt < 0.15:
                if s[i] > s[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)

    events = []
    for i in peaks:
        p = s[i]
        r = 1
        while i - r >= 0 and s[i - r] > 0.25 * p and r < 8:
            r += 1
        f = 1
        while i + f < len(s) and s[i + f] > 0.15 * p and f < 8:
            f += 1
        d = np.array([vx[i], vy[i]]) / (p + 1e-9)
        after = [np.dot([vx[j], vy[j]], d) for j in range(i + 1, min(i + 7, len(s)))]
        overshoot = bool(after) and bool(min(after) < -0.10 * p)
        events.append(
            dict(i=i, peak=float(p), rise=r, fall=f,
                 instant=(r <= 2 and f <= 2), overshoot=overshoot,
                 stop_ratio=float(s[i + 1] / p) if i + 1 < len(s) else 0.0)
        )
    return s, events


def detect_overlay_boxes(path, max_sec, progress):
    """순수 빨강/초록 속이 빈 사각형(ESP 박스 의심)이 있는 프레임 비율."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = min(total / fps, max_sec) if max_sec else total / fps
    hits = checked = 0
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (640, int(h * 640 / w)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, (0, 200, 200), (8, 255, 255))
                | cv2.inRange(hsv, (172, 200, 200), (180, 255, 255))
                | cv2.inRange(hsv, (52, 200, 200), (68, 255, 255)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = False
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 12 or bh < 24 or not (1.3 <= bh / bw <= 4.5):
                continue
            fill = cv2.countNonZero(mask[y:y + bh, x:x + bw]) / (bw * bh)
            approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
            if 0.04 < fill < 0.3 and len(approx) == 4:
                found = True
                break
        hits += found
        checked += 1
        t += 0.5
        progress.progress(min(t / max(dur, 1e-9), 1.0))
    cap.release()
    return hits / checked if checked else 0.0


def grab_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def compute_score(events, minutes, esp_ratio, use_esp):
    n = len(events)
    info = dict(n=n, inst=0, inst_ratio=0.0, no_os=0.0, rate=0.0, cv=None)
    if n == 0:
        return 0, info
    inst = [e for e in events if e["instant"]]
    info["inst"] = len(inst)
    info["inst_ratio"] = len(inst) / n
    info["no_os"] = float(np.mean([not e["overshoot"] for e in inst])) if inst else 0.0
    info["rate"] = len(inst) / max(minutes, 1 / 60)
    peaks = np.array([e["peak"] for e in inst])
    if len(peaks) >= 5:
        info["cv"] = float(peaks.std() / peaks.mean())
    uni = max(0.0, 1 - info["cv"] / 0.5) if info["cv"] is not None else 0.0
    raw = (35 * info["inst_ratio"] + 25 * info["no_os"]
           + 15 * min(info["rate"] / 15, 1) + 10 * uni)
    conf = min(1.0, n / 8)
    total = raw * conf + (15 * min(esp_ratio / 0.4, 1) if use_esp else 0)
    return int(round(min(total, 100))), info


def verdict(score):
    if score < 25:
        return "🟢 정상 범위", "success"
    if score < 50:
        return "🟡 주의 (약한 신호)", "warning"
    if score < 75:
        return "🟠 의심", "warning"
    return "🔴 강한 의심", "error"


def ensure_module(mod, pip_name, pre=None):
    """모듈이 없으면 지금 실행 중인 파이썬(sys.executable)에 자동 설치 후 import."""
    import importlib
    try:
        return importlib.import_module(mod)
    except ImportError:
        pass
    # pre가 있으면: 의존성은 따로 깔고 본 패키지는 --no-deps로 설치 (opencv 중복 설치/덮어쓰기 충돌 방지)
    if pre:
        subprocess.run([sys.executable, "-m", "pip", "install", *pre], capture_output=True, text=True)
    cmd = [sys.executable, "-m", "pip", "install", "-U", pip_name] + (["--no-deps"] if pre else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and "externally-managed" in (r.stderr or ""):
        # 리눅스/맥 시스템 파이썬(PEP 668)이면 사용자 영역에만 설치하도록 재시도
        r = subprocess.run(cmd + ["--user", "--break-system-packages"], capture_output=True, text=True)
    try:  # 방금 만들어진 사용자 site-packages 폴더를 현재 프로세스 경로에 추가
        import site
        site.addsitedir(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    importlib.invalidate_caches()
    try:
        return importlib.import_module(mod)
    except ImportError:
        raise RuntimeError(
            f"{pip_name} 자동 설치에 실패했어요. 터미널에서 `{sys.executable} -m pip install -U \"{pip_name}\"` "
            f"를 실행한 뒤 앱을 다시 시작하세요. {(r.stderr or '')[-200:]}")


class TooLong(Exception):
    pass


def js_runtime_status():
    return {n: shutil.which(n) for n in ("deno", "node", "bun", "qjs")}


def hint_for_error(msg):
    m = msg.lower()
    hints = []
    if any(k in m for k in ("javascript runtime", "only images are available",
                            "requested format is not available", "403", "nsig", "signature")):
        hints.append("유튜브는 yt-dlp가 **JS 런타임(Deno 권장)**을 요구해요. "
                     "Deno 설치(Windows: `winget install DenoLand.Deno`, mac: `brew install deno`, "
                     "Linux: `curl -fsSL https://deno.land/install.sh | sh`) 후 "
                     "`pip install -U 'yt-dlp[default]'` 하고 **터미널을 새로 열어 앱을 다시 실행**하세요.")
    if any(k in m for k in ("sign in", "not a bot", "cookies", "login")):
        hints.append("로그인/봇 확인이 필요한 영상이에요. 사이드바 '🍪 다운로드 문제 해결'에서 브라우저 쿠키를 선택하거나 cookies.txt를 올려보세요.")
    if "429" in m or "too many requests" in m:
        hints.append("요청이 너무 많아 잠시 차단됐어요. 몇 분 뒤 다시 시도하세요.")
    if "unsupported url" in m:
        hints.append("지원하지 않는 URL이에요. 영상 페이지 주소나 직접 .mp4 링크인지 확인하세요.")
    if "ffmpeg" in m:
        hints.append("ffmpeg가 필요해요. 설치 후 PATH에 추가하세요 (예: `winget install ffmpeg`, `brew install ffmpeg`).")
    if not hints:
        hints.append("`pip install -U 'yt-dlp[default]'`로 최신 버전으로 올린 뒤 다시 시도해 보세요.")
    return hints


def download_video(url, max_min=30, cookie_browser=None, cookie_bytes=None):
    """yt-dlp로 URL의 영상을 임시 폴더에 내려받는다. 클라이언트를 바꿔가며 최대 3번 시도."""
    yt_dlp = ensure_module("yt_dlp", "yt-dlp[default]")

    tmpdir = tempfile.mkdtemp()
    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        fmt = ("bv*[height<=720][fps>=60][vcodec^=avc1]+ba/bv*[height<=720][vcodec^=avc1]+ba/"
               "bv*[height<=720]+ba/b[height<=720]/b")
    else:
        fmt = "b[ext=mp4][height<=720][fps>=60]/b[ext=mp4][height<=720]/b[height<=720]/b"
    opts = {
        "format": fmt,
        "outtmpl": os.path.join(tmpdir, "video.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "max_filesize": 600 * 1024 * 1024,
    }
    if has_ffmpeg:
        opts["merge_output_format"] = "mp4"
    if cookie_bytes:
        cpath = os.path.join(tmpdir, "cookies.txt")
        with open(cpath, "wb") as fh:
            fh.write(cookie_bytes)
        opts["cookiefile"] = cpath
    elif cookie_browser:
        opts["cookiesfrombrowser"] = (cookie_browser,)

    rt = js_runtime_status()
    variants = [{}, {"extractor_args": {"youtube": {"player_client": ["android_vr"]}}}]
    if not rt["deno"] and rt["node"]:
        variants.insert(0, {"js_runtimes": {"node": {}}})

    last_err = None
    for extra in variants:
        for f in os.listdir(tmpdir):
            if f != "cookies.txt":
                try:
                    os.unlink(os.path.join(tmpdir, f))
                except OSError:
                    pass
        try:
            with yt_dlp.YoutubeDL({**opts, **extra}) as ydl:
                info = ydl.extract_info(url, download=False)
                dur = info.get("duration")
                if dur and dur > max_min * 60:
                    raise TooLong(f"영상이 너무 길어요 ({dur/60:.0f}분). {max_min}분 이하만 지원합니다.")
                ydl.download([url])
            files = [f for f in os.listdir(tmpdir) if f != "cookies.txt" and not f.endswith((".part", ".ytdl"))]
            if files:
                return os.path.join(tmpdir, files[0]), tmpdir
            last_err = RuntimeError("영상 파일을 받지 못했어요 (용량 제한 초과 가능).")
        except TooLong:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        except Exception as ex:  # noqa: BLE001
            last_err = ex
    shutil.rmtree(tmpdir, ignore_errors=True)
    raise last_err


def _merge_events(cands, min_gap):
    """(time, k, strength) 후보를 시간 간격으로 합치고 가장 강한 것만 남김."""
    out, last = [], -1e9
    for t, k, st_ in cands:
        if t - last < min_gap and out:
            if st_ > out[-1]["strength"]:
                out[-1] = dict(k=k, strength=st_)
        else:
            out.append(dict(k=k, strength=st_))
        last = t
    return out


def detect_teleports(extra, dt, thr):
    """페이드/섬광 없이 화면 전체가 한 프레임 만에 완전히 바뀌는 순간 (순간이동·컷 후보)."""
    h, b, bd, nc = extra["hcorr"], extra["bright"], extra["bdiff"], extra["ncc"]
    cands = [(k * dt, k, float(max(0.0, 1 - min(h[k], 3 * nc[k])))) for k in range(len(h))
             if (h[k] < thr or nc[k] < thr / 3) and 25 < b[k] < 230 and bd[k] < 50]
    return _merge_events(cands, 1.0)


def detect_popins(extra, s, dt, thr):
    """카메라 이동은 적은데 화면 일부가 갑자기 크게 바뀌는 순간 (소환·오브젝트 팝인 후보)."""
    loc, nc, bd = extra["local"], extra["ncc"], extra["bdiff"]
    base = float(np.median(loc)) if len(loc) else 0.0
    lim = max(thr, 4 * base)
    cands = [(k * dt, k, float(loc[k])) for k in range(len(loc))
             if loc[k] >= lim and nc[k] >= 0.4 and s[k] < 1.5 and bd[k] < 40]
    return _merge_events(cands, 0.7)


def show_candidates(title, evs, idxs, fps, stride, path, minutes):
    st.markdown(f"### {title}")
    rate = len(evs) / max(minutes, 1 / 60)
    c1, c2 = st.columns(2)
    c1.metric("후보 구간", len(evs))
    c2.metric("분당", f"{rate:.1f}")
    if not evs:
        st.success("뚜렷한 후보가 없어요.")
        return
    top = sorted(evs, key=lambda e: -e["strength"])[:6]
    cols = st.columns(len(top))
    for col, e in zip(cols, top):
        fi = int(idxs[e["k"]])
        a, b = grab_frame(path, fi - stride), grab_frame(path, fi)
        with col:
            st.caption(f"{fi / fps:.2f}s · 강도 {e['strength']:.2f}")
            if a is not None:
                st.image(a, caption="직전", use_container_width=True)
            if b is not None:
                st.image(b, caption="직후", use_container_width=True)


GAME_PROFILES = {
    "자동 식별 (AI)": dict(aim=True, esp=True, tele=True, spawn=True),
    "발로란트": dict(aim=True, esp=True, tele=False, spawn=False),
    "로블록스 라이벌": dict(aim=True, esp=True, tele=True, spawn=False),
    "지오메트리 대시": dict(aim=False, esp=False, tele=False, spawn=False),
    "포켓몬 고": dict(aim=False, esp=False, tele=True, spawn=False),
}

FORENSIC_PROMPT = """# Role & Purpose
당신은 게임 안티치트 데이터 포렌식 전문가입니다. 제시되는 플레이 영상 프레임과 자동 계산된 영상 텔레메트리를, 지정된 4개 게임(라이벌, 발로란트, 지메, 포고)의 고유 핵 메커니즘에 맞추어 과학적으로 판독하세요.

# Target Games & Detection Rules

[1. 발로란트 (Valorant)]
- 에임봇/트리거봇: 미세 손떨림 및 가속도 유무, 적 노출 즉시 보정(Snap) 여부, 히트박스(헤드) 강제 락온 및 오버슈팅 생략 여부
- ESP/월핵: 벽 뒤의 적을 시야 노출 전 예샷/사선 추적하는 행위 (사운드 플레이 한계 초과)
- 반동/스프레이: 총기 반동(Recoil) 패턴이 완전 정지 상태이거나 기계적으로 일정하게 상쇄되는지 여부

[2. 로블록스 라이벌 (Roblox Rivals)]
- Silent Aim: 화면상의 에임 십자선은 적을 향하지 않는데 총알 투사체가 적에게 곡선으로 유도되거나 꺾여 맞는지 여부
- 히트박스 확장 (Hitbox Expander): 허공을 사격했음에도 데미지 판정이 들어가는지 여부
- 속도/이동 핵: 일반 캐릭터 이동 속도 범위를 초과하는 이상 위치 이동 또는 공중 비행 여부

[3. 지오메트리 대시 (Geometry Dash)]
- 노클립 (Noclip): 톱니바퀴, 가시, 블록 등의 장애물 레이어에 히트박스가 물리적으로 겹침(overlapping)되었음에도 사망하지 않고 통과하는지 여부
- 스피드핵 / Bot: 음악의 BPM과 입체의 움직임 타이밍 불일치, 프레임 단위(1/60초 이하)의 기계적·완벽한 입력 간격 유지 여부
- 자막/클리어 데이터: Cheat Indicator 표시 여부 또는 일시정지 후 프레임 스킵 현상

[4. 포켓몬 고 (Pokemon GO)]
- GPS 스푸핑 / 위치 조작: 물리적으로 불가능한 짧은 시간 내 시·공간 이동(쿨다운 법칙 위반), 캐릭터 위치 텔레포트 및 지도 텍스처 로딩 지연
- 엑설런트/패스트 캐치 봇: 원의 크기 변화와 무관하게 100% 확률로 Excellent 찌르기 판정 달성, 모션 애니메이션 강제 스킵 패턴
- 스푸핑 앱 UI 흔적: 화면에 뜬 조이스틱 오버레이, 쿨다운 타이머, 포획 전 IV/개체값 표시, 주변 레이더 같은 비공식 오버레이

[공통: 치트 프로그램 흔적 — 직접 증거]
- 익스큐터/인젝터/스크립트 허브 창, Execute·Inject·Attach 버튼, Lua 스크립트 코드(loadstring, game:HttpGet 등), Aimbot/ESP/Silent Aim/Kill Aura/Fly/Speed 같은 토글이 있는 치트 메뉴 GUI, 스푸핑 앱 오버레이가 영상에 보이면 그 자체로 핵(치트) 사용의 직접 증거입니다.

# Analysis Instructions
1. 전달받은 영상/데이터가 위 4개 중 어느 게임인지 먼저 식별하십시오. (사용자가 게임을 지정했다면 그것을 우선하되, 영상과 명백히 다르면 지적하십시오.)
2. 해당 게임의 판독 기준을 적용하여 이상징후를 분석하십시오.
3. 단순 실력자(고감도 유저, 고수 플릭 shot, 정교한 셋업)의 플레이와 실제 핵 프로그램 사용 패턴을 명확히 구분하십시오.

# Evidence Rules (반드시 지킬 것)
- 근거는 첨부된 프레임과 텔레메트리에 있는 것만 사용하세요. 보이지 않거나 계산되지 않은 항목은 지어내지 말고 '확인 불가'라고 쓰세요.
- 텔레메트리는 영상 기반 휴리스틱이며 오디오·입력 로그·서버 데이터가 없습니다. 영상 편집 컷, 리스폰, 로딩, 낮은 fps, 압축 화질이 오탐을 만들 수 있음을 감안하세요.
- '확정적'은 치트 UI/인디케이터/익스큐터/스크립트 GUI처럼 영상에 직접 찍힌 증거가 있거나, 텔레메트리의 cheat_ui_ocr_hits에 strong 항목이 있을 때만 사용하세요. 통계적 이상만으로는 '매우 높음'까지만 사용하세요.
- 타임라인에는 실제로 제공된 시간/프레임만 인용하세요.

# Output Format
- [대상 게임]: (라이벌 / 발로란트 / 지메 / 포고 중 택1)
- [최종 의심도]: (의심 없음 / 낮음 / 보통 / 매우 높음 / 확정적)
- [주요 감지 포인트]: 해당 게임 기준에 부합하는 결정적 이상 징후 (최대 3개)
- [구간별 상세 포렌식 분석]:
  - 타임라인(또는 프레임 번호) | 관찰된 비정상 동작 | 핵 가능성 및 이유
"""


def _longest_run(mask, dt):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * dt


def build_telemetry(game, fps, dt, stride, s, events, info, score, teleports, popins, esp_ratio, extra, idxs, ui_hits=None):
    r = lambda x, n=3: round(float(x), n)
    tsec = lambda k: r(idxs[k] / fps, 2)
    mad = extra["mad"]
    moving = s[s > 0.1]
    med = float(np.median(moving)) if len(moving) else 0.0
    stable = (np.abs(s - med) < 0.1 * med) & (s > 0.1) if med > 0 else np.zeros(len(s), dtype=bool)
    return {
        "game_selected": game,
        "video": {"fps": r(fps, 1), "analyzed_seconds": r(len(s) * dt, 1), "frame_stride": stride},
        "aim_heuristics": {
            "heuristic_score_0_100": score,
            "snap_events": info["n"],
            "instant_snap_stops": info["inst"],
            "instant_snap_no_overshoot_ratio": r(info["no_os"]),
            "instant_snaps_per_min": r(info["rate"], 2),
            "snap_peak_speed_cv": None if info["cv"] is None else r(info["cv"]),
            "max_camera_speed_widths_per_sec": r(s.max()) if len(s) else 0,
            "top_snaps": [dict(t=tsec(e["i"]), peak=r(e["peak"], 2), rise_frames=e["rise"],
                               fall_frames=e["fall"], instant=e["instant"], overshoot=e["overshoot"])
                          for e in sorted(events, key=lambda e: -e["peak"])[:8]],
        },
        "teleport_candidates": [dict(t=tsec(e["k"]), strength=r(e["strength"], 2))
                                for e in sorted(teleports, key=lambda e: -e["strength"])[:10]],
        "popin_candidates": [dict(t=tsec(e["k"]), strength=r(e["strength"], 2))
                             for e in sorted(popins, key=lambda e: -e["strength"])[:10]],
        "esp_like_box_frame_ratio": None if esp_ratio is None else r(esp_ratio),
        "frame_integrity": {
            "frozen_or_duplicate_frame_ratio": r((mad < 0.15).mean()) if len(mad) else 0,
            "longest_freeze_seconds": r(_longest_run(mad < 0.15, dt), 2),
        },
        "camera_motion": {
            "median_speed_when_moving": r(med),
            "p95_speed": r(np.percentile(s, 95)) if len(s) else 0,
            "speed_cv_when_moving": r(moving.std() / med) if med > 0 else None,
            "longest_constant_speed_run_seconds": r(_longest_run(stable, dt), 2),
        },
        "cheat_ui_ocr_hits": [dict(t=round(h["t"], 2), kind=h["kind"], terms=h["terms"]) for h in (ui_hits or [])[:8]],
        "notes": "영상 기반 휴리스틱입니다. 오디오/입력 로그/서버 데이터 없음. 컷·리스폰·로딩·저fps가 오탐을 만들 수 있음.",
    }


def pick_frames(path, idxs, fps, stride, events, teleports, popins, n=14):
    picks = []
    for e in sorted([e for e in events if e["instant"]], key=lambda e: -e["peak"])[:2]:
        fi = int(idxs[e["i"]])
        picks += [(fi - 2 * stride, "스냅 직전"), (fi + 2 * stride, "스냅 직후")]
    for lab, evs in (("순간이동 후보", teleports), ("팝인 후보", popins)):
        for e in sorted(evs, key=lambda e: -e["strength"])[:2]:
            fi = int(idxs[e["k"]])
            picks += [(fi - stride, lab + " 직전"), (fi, lab + " 직후")]
    total = int(idxs[-1]) if len(idxs) else 0
    need = max(0, n - len(picks))
    picks += [(int(total * (q + 1) / (need + 1)), "구간 샘플") for q in range(need)]
    out = []
    for fi, lab in sorted(set(picks)):
        img = grab_frame(path, fi)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w > 768:
            img = cv2.resize(img, (768, int(h * 768 / w)))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append((fi / fps, lab, base64.b64encode(buf).decode()))
    return out


def run_llm_report(api_key, model, game, telemetry, frames):
    anthropic = ensure_module("anthropic", "anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    content = []
    for ts, lab, b64 in frames:
        content.append({"type": "text", "text": f"[프레임 t={ts:.2f}s | {lab}]"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    hint = "사용자가 게임을 지정하지 않았습니다. 직접 식별하세요." if game.startswith("자동") else f"사용자가 지정한 게임: {game}"
    content.append({"type": "text", "text":
                    f"{hint}\n\n자동 계산된 영상 텔레메트리(JSON):\n{json.dumps(telemetry, ensure_ascii=False, indent=1, default=str)}\n\n"
                    "위 규칙과 출력 형식에 맞춰 분석하세요."})
    resp = client.messages.create(model=model, max_tokens=2500, system=FORENSIC_PROMPT,
                                  messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in resp.content if b.type == "text")


UI_STRONG = [
    "executor", "exploit", "injector", "inject", "script hub", "scripthub", "loadstring", "httpget", "getgenv",
    "aimbot", "aim bot", "silent aim", "triggerbot", "trigger bot", "wallhack", "kill aura", "killaura",
    "hitbox expander", "krnl", "synapse x", "fluxus", "solara", "jjsploit", "madium", "divinex", "cyrex",
    "byfron", "keyless", "rayfield", "megahack", "mega hack", "speedhack", "speed hack", "noclip", "no clip",
    "cheat indicator", "pgsharp", "fake gps", "gps spoof",
]
UI_WEAK = ["esp", "fly", "speed", "aim", "teleport", "auto farm", "infinite jump", "god mode",
           "hitbox", "fov", "walkspeed", "jumppower", "bhop", "autoshoot"]

_OCR = {}


def detect_cheat_ui(path, max_sec, step_sec, progress):
    """OCR로 익스큐터/스크립트 GUI/치트 메뉴 글자를 찾는다. strong=키워드 1개로 확정, weak=토글 메뉴로 보이는 3개 이상."""
    import re
    if "engine" not in _OCR:
        mod = ensure_module("rapidocr_onnxruntime", "rapidocr-onnxruntime",
                            pre=["onnxruntime", "pyclipper", "shapely", "pyyaml", "pillow", "six"])
        _OCR["engine"] = mod.RapidOCR()
    ocr = _OCR["engine"]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = min(total / fps, max_sec) if max_sec else total / fps
    hits, t = [], 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 1600:
            frame = cv2.resize(frame, (1600, int(h * 1600 / w)))
        res, _ = ocr(frame)
        blob = " ".join(r[1] for r in res).lower() if res else ""
        flat = re.sub(r"[^a-z0-9]", "", blob)
        strong = [k for k in UI_STRONG if re.sub(r"[^a-z0-9]", "", k) in flat]
        weak = [k for k in UI_WEAK if re.search(r"\b" + re.escape(k) + r"\b", blob)]
        if strong:
            hits.append(dict(t=t, kind="strong", terms=strong[:5], text=blob[:140]))
        elif len(weak) >= 3:
            hits.append(dict(t=t, kind="weak", terms=weak[:6], text=blob[:140]))
        t += step_sec
        progress.progress(min(t / max(dur, 1e-9), 1.0))
    cap.release()
    return hits[:30]


# ---------------------------------------------------------------- UI
st.title("🕵️ 치트 의심 영상 분석기")
st.caption("발로란트 · 로블록스 라이벌 · 지오메트리 대시 · 포켓몬 고 플레이 영상을 게임별 기준으로 분석합니다.")
st.warning(
    "⚠️ 영상만으로는 핵 사용을 **확정할 수 없습니다.** 이 도구는 의심 신호를 찾는 보조 수단이며, "
    "제재/비난의 단독 근거로 쓰면 안 됩니다. (프로 선수의 빠른 플릭, 낮은 fps, 압축 화질도 영향을 줍니다.)"
)

with st.sidebar:
    st.header("⚙️ 설정")
    game = st.selectbox("🎮 게임", list(GAME_PROFILES), help="게임에 맞는 탐지기가 기본으로 켜져요. 모르면 자동 식별(AI)을 고르세요.")
    prof = GAME_PROFILES[game]
    show_aim = prof["aim"]
    max_sec = st.slider("분석할 최대 길이(초)", 10, 300, 90, 10)
    stride = st.select_slider("프레임 간격 (1=전 프레임, 정밀/느림)", [1, 2, 3], value=1)
    thr = st.slider("스냅 판정 속도 (화면폭/초)", 1.0, 10.0, 3.0, 0.5,
                    help="이 속도 이상으로 시점이 휙 돌아가면 '스냅' 후보로 봅니다. 감도가 높은 게임이면 올리세요.")
    use_esp = st.checkbox("ESP 박스 오버레이 탐지 (가중치 낮음)", value=prof["esp"], key=f"esp_{game}",
                          help="체력바/킬로그와 헷갈려 오탐이 있을 수 있어요.")
    st.markdown("---")
    use_tele = st.checkbox("⚡ 순간이동 의심 탐지", value=prof["tele"], key=f"tele_{game}",
                           help="페이드 없이 장면이 통째로 바뀌는 순간을 찾아요. 리스폰/영상 컷도 걸릴 수 있어요.")
    tele_thr = st.slider("장면 유사도 임계값", 0.1, 0.8, 0.45, 0.05,
                         help="낮을수록 더 확실히 바뀐 순간만 잡아요.")
    use_spawn = st.checkbox("👾 소환/팝인 의심 탐지", value=prof["spawn"], key=f"spawn_{game}",
                            help="카메라는 가만한데 화면 일부가 갑자기 바뀌는 순간. 폭발·이펙트도 걸릴 수 있어요.")
    spawn_thr = st.slider("국소 변화 임계값", 0.03, 0.4, 0.08, 0.01,
                          help="낮을수록 작은 변화도 잡아요(오탐↑). 화면 많은 부분이 바뀔 때만 보려면 올리세요.")
    with st.expander("🍪 다운로드 문제 해결 (쿠키)"):
        cookie_browser = st.selectbox("브라우저 쿠키 사용", ["사용 안 함", "chrome", "firefox", "edge", "brave", "safari"])
        cookie_file = st.file_uploader("또는 cookies.txt 업로드", type=["txt"])
    st.markdown("---")
    use_ui = st.checkbox("🧾 치트 UI/스크립트 글자 인식 (OCR)", value=True,
                         help="익스큐터 창, 스크립트 코드, Aimbot/ESP 토글 메뉴 같은 글자가 화면에 보이면 핵으로 판정해요. 처음엔 OCR 모델 설치로 시간이 걸려요.")
    ui_step = st.slider("OCR 검사 간격(초)", 1, 5, 2) if use_ui else 2
    use_ai = st.checkbox("🤖 AI 포렌식 리포트 (Claude)", value=False,
                         help="프레임 약 14장과 분석 수치를 Anthropic API로 보내 게임별 기준 리포트를 작성해요. API 키 필요, 비용 발생.")
    api_key = st.text_input("Anthropic API 키", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password") if use_ai else ""
    ai_model = st.text_input("모델", value="claude-sonnet-5-5") if use_ai else ""
    st.markdown("---")
    st.markdown("**권장 영상**: 60fps 이상, 1인칭 시점, 조준 장면이 많은 구간")

tab_url, tab_file = st.tabs(["🔗 URL로 분석", "📁 파일 업로드"])
with tab_url:
    url = st.text_input("영상 URL", placeholder="https://www.youtube.com/watch?v=...  또는 직접 .mp4 링크")
    st.caption("유튜브·트위치 클립·X 등 yt-dlp가 지원하는 사이트와 직접 영상 링크가 가능해요. 비공개/로그인 필요 영상은 안 돼요.")
    go_url = st.button("🔍 URL 분석 시작", type="primary", key="go_url")
with tab_file:
    file = st.file_uploader("영상 업로드", type=["mp4", "mov", "avi", "mkv", "webm"])
    if file:
        st.video(file)
    go_file = st.button("🔍 파일 분석 시작", type="primary", key="go_file", disabled=file is None)

path, cleanup_dir = None, None
if go_url:
    if not url.strip():
        st.error("URL을 입력해 주세요.")
        st.stop()
    try:
        with st.spinner("영상 다운로드 중… (길이/화질에 따라 시간이 걸려요)"):
            path, cleanup_dir = download_video(
                url.strip(),
                cookie_browser=None if cookie_browser == "사용 안 함" else cookie_browser,
                cookie_bytes=cookie_file.getvalue() if cookie_file else None)
        st.video(path)
    except Exception as ex:
        st.error(f"다운로드 실패: {ex}")
        for h in hint_for_error(str(ex)):
            st.info(h)
        rt = js_runtime_status()
        if not any(rt.values()):
            st.warning("PATH에서 JS 런타임(deno/node)을 못 찾았어요. 유튜브 다운로드 실패의 가장 흔한 원인이에요.")
        st.stop()
elif go_file and file:
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getbuffer())
        path = tmp.name

if path:
    try:
        st.write("**1/2 조준 움직임 분석 중…**")
        bar = st.progress(0.0)
        fps, dt, vx, vy, idxs, extra = analyze_motion(path, stride, max_sec, bar)
        if len(vx) < 10:
            st.error("분석할 프레임이 너무 적어요. 영상을 확인해 주세요.")
            st.stop()
        if fps < 50:
            st.warning(f"영상이 {fps:.0f}fps예요. 60fps 미만이면 정상 플릭도 '즉시 스냅'처럼 보여 오탐이 늘어요.")
        s, events = detect_snaps(vx, vy, dt, thr)
        teleports = detect_teleports(extra, dt, tele_thr) if use_tele else []
        popins = detect_popins(extra, s, dt, spawn_thr) if use_spawn else []
        popins = [p for p in popins if all(abs(p["k"] - t["k"]) * dt > 1.0 for t in teleports)]
        minutes = len(vx) * dt / 60

        esp_ratio = 0.0
        if use_esp:
            st.write("**2/2 오버레이 박스 탐지 중…**")
            bar2 = st.progress(0.0)
            esp_ratio = detect_overlay_boxes(path, max_sec, bar2)

        ui_hits = []
        if use_ui:
            st.write("**치트 UI/스크립트 글자(OCR) 찾는 중…** 처음 한 번은 모델 설치로 오래 걸려요.")
            bar3 = st.progress(0.0)
            try:
                ui_hits = detect_cheat_ui(path, max_sec, ui_step, bar3)
            except Exception as ex:  # noqa: BLE001
                st.warning(f"OCR 탐지를 건너뛰었어요: {ex}")
        ui_strong = [h for h in ui_hits if h["kind"] == "strong"]

        score, info = compute_score(events, minutes, esp_ratio, use_esp)
        if ui_strong:
            score = 100
        elif ui_hits:
            score = max(score, 85)
        label, level = verdict(score)
        if ui_strong:
            label = "🔴 확정 — 치트 UI/스크립트 감지"

        if ui_hits:
            if ui_strong:
                st.error(f"🚨 **핵 사용으로 판정** — 익스큐터/스크립트/치트 메뉴 흔적 {len(ui_hits)}건이 화면에서 발견됐어요.")
            else:
                st.warning(f"⚠️ 핵 메뉴로 보이는 화면 {len(ui_hits)}건 — 토글형 치트 메뉴일 가능성이 높아요. 점수를 85 이상으로 올렸어요.")
            st.dataframe(pd.DataFrame([{
                "시간(초)": round(h["t"], 1), "구분": "확정급" if h["kind"] == "strong" else "의심",
                "감지 단어": ", ".join(h["terms"]), "화면 글자(일부)": h["text"]} for h in ui_hits]),
                use_container_width=True)
            ucols = st.columns(min(len(ui_hits), 4))
            for col, h in zip(ucols, sorted(ui_hits, key=lambda h: h["kind"] != "strong")[:4]):
                img = grab_frame(path, int(h["t"] * fps))
                if img is not None:
                    col.image(img, caption=f"{h['t']:.1f}s · {', '.join(h['terms'][:3])}", use_container_width=True)
            st.caption("※ 핵 설명/리뷰 영상처럼 치트 화면이 '소개' 목적으로 나와도 글자만 보고 판정하기 때문에 똑같이 걸려요.")
        if show_aim:
            st.markdown("## 📊 결과")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("의심 점수", f"{score} / 100")
                getattr(st, level)(label)
                if info["n"] < 5:
                    st.info("스냅 표본이 5개 미만이라 신뢰도가 낮습니다. 조준 장면이 더 많은 영상을 써보세요.")
            with c2:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("스냅 이벤트", info["n"])
                m2.metric("즉시 스냅-정지", info["inst"])
                m3.metric("오버슈트 없음", f"{info['no_os']*100:.0f}%")
                m4.metric("분당 즉시 스냅", f"{info['rate']:.1f}")
                m5, m6 = st.columns(2)
                m5.metric("스냅 속도 편차(CV)", "-" if info["cv"] is None else f"{info['cv']:.2f}",
                          help="낮을수록 매번 같은 속도로 꽂힘 → 기계적")
                m6.metric("ESP 박스 의심 프레임", f"{esp_ratio*100:.0f}%" if use_esp else "미사용")

            st.markdown("### 📈 조준 이동 속도 타임라인")
            t = idxs / fps
            st.line_chart(pd.DataFrame({"속도(화면폭/초)": s}, index=np.round(t, 2)))

            if events:
                df = pd.DataFrame([{
                    "시간(초)": round(idxs[e["i"]] / fps, 2),
                    "피크 속도": round(e["peak"], 2),
                    "가속 프레임": e["rise"], "감속 프레임": e["fall"],
                    "즉시 스냅-정지": "✅" if e["instant"] else "",
                    "오버슈트": "있음" if e["overshoot"] else "없음",
                } for e in events])
                st.markdown("### 🎯 스냅 이벤트 목록")
                st.dataframe(df, use_container_width=True)

                sus = sorted([e for e in events if e["instant"] and not e["overshoot"]],
                             key=lambda e: -e["peak"])[:4]
                if sus:
                    st.markdown("### 🖼️ 가장 수상한 순간 (직전 → 직후)")
                    cols = st.columns(len(sus))
                    for col, e in zip(cols, sus):
                        fi = int(idxs[e["i"]])
                        a = grab_frame(path, fi - 2 * stride)
                        b = grab_frame(path, fi + 3 * stride)
                        with col:
                            st.caption(f"{fi / fps:.2f}s · 피크 {e['peak']:.1f}")
                            if a is not None:
                                st.image(a, use_container_width=True)
                            if b is not None:
                                st.image(b, use_container_width=True)

        if use_tele or use_spawn:
            st.markdown("## 🧭 순간이동 / 소환(팝인) 의심 구간")
            st.caption("영상 편집 컷, 리스폰, 로딩, 폭발 이펙트와 구분이 어려워서 '사람이 눈으로 확인할 후보'로만 보여줘요.")
            if use_tele:
                show_candidates("⚡ 순간이동 의심", teleports, idxs, fps, stride, path, minutes)
            if use_spawn:
                show_candidates("👾 소환/오브젝트 팝인 의심", popins, idxs, fps, stride, path, minutes)
            with st.expander("순간이동·소환핵 탐지의 한계"):
                st.markdown(
                    "- 영상으론 **위치 좌표나 스폰 권한**을 알 수 없어서, 화면이 비정상적으로 바뀐 순간만 후보로 뽑아요.\n"
                    "- 정상적인 리스폰·포탈·컷신·편집 컷·대형 이펙트도 걸려요. 반드시 직전/직후 프레임을 눈으로 확인하세요.\n"
                    "- 이 두 종류는 **서버 쪽 검증**(이동 속도/거리 검사, 스폰 요청 권한 로그)이 훨씬 정확해요. 게임 서버가 있다면 그쪽 로그를 우선하세요."
                )
        telemetry = build_telemetry(game, fps, dt, stride, s, events, info, score, teleports, popins,
                                    esp_ratio if use_esp else None, extra, idxs, ui_hits)
        if not show_aim:
            st.markdown("## 🎞️ 프레임·이동 보조 지표")
            q1, q2, q3 = st.columns(3)
            q1.metric("정지/중복 프레임 비율", f"{telemetry['frame_integrity']['frozen_or_duplicate_frame_ratio']*100:.0f}%",
                      help="일시정지 후 프레임 스킵, 렉, 편집 흔적일 수 있어요.")
            q2.metric("최장 정지 구간", f"{telemetry['frame_integrity']['longest_freeze_seconds']:.1f}s")
            q3.metric("최장 등속 이동", f"{telemetry['camera_motion']['longest_constant_speed_run_seconds']:.1f}s",
                      help="사람 입력으로는 보기 드문 완벽한 등속 이동(오토워크 류)을 가늠하는 값이에요.")
        with st.expander("📦 분석 텔레메트리 (JSON)"):
            st.json(telemetry)
        if use_ai:
            st.markdown("## 🧾 AI 포렌식 리포트")
            if not api_key:
                st.warning("사이드바에 Anthropic API 키를 입력하세요.")
            else:
                try:
                    with st.spinner("프레임을 고르고 리포트를 작성하는 중…"):
                        frames = pick_frames(path, idxs, fps, stride, events, teleports, popins)
                        report = run_llm_report(api_key, ai_model, game, telemetry, frames)
                    st.markdown(report)
                    with st.expander("AI에 보낸 프레임"):
                        fc = st.columns(4)
                        for n_, (ts, lab, b64) in enumerate(frames):
                            fc[n_ % 4].image(base64.b64decode(b64), caption=f"{ts:.2f}s · {lab}")
                except Exception as ex:  # noqa: BLE001
                    st.error(f"AI 리포트 실패: {ex}")
                    if "anthropic" in str(ex).lower() and "module" in str(ex).lower():
                        st.info("`pip install anthropic` 후 다시 실행하세요.")
        with st.expander("판정 방식 / 한계"):
            st.markdown(
                "- **즉시 스냅-정지**: 시점이 1~2프레임 만에 가속되고 1~2프레임 만에 멈추는 움직임. 사람 손은 관성 때문에 보통 감속 구간이 있습니다.\n"
                "- **오버슈트 없음**: 스냅 후 목표를 지나쳤다가 되돌아오는 보정이 없는 비율.\n"
                "- **속도 편차(CV)**: 스냅 피크 속도가 얼마나 균일한지. 에임봇은 설정값대로 일정해지는 경향이 있습니다.\n"
                "- **ESP 박스**: 순수 빨강/초록의 속이 빈 사각형이 사람 비율로 나타나는 프레임 비율.\n\n"
                "**한계**: 월핵·레이더·트리거봇·스크립트 일부는 영상 화면만으론 못 잡습니다. "
                "30fps 이하 영상은 정상 플릭도 '즉시 스냅'처럼 보이고, 무기 모델·HUD·영상 편집이 오탐을 만들 수 있습니다."
            )
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)
else:
    st.info("URL을 붙여넣거나 영상을 업로드한 뒤 분석을 시작하세요.")
