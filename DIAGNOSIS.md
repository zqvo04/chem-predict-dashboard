# 현재 상태 진단 — 왜 파이프라인이 정적인가, 그리고 타깃 패밀리 일반화는 어디까지 왔는가

**Status: 진단(측정).** 이 문서의 모든 수치는 커밋된 자산(`assets/`)과 이 저장소의
코드로 이 체크아웃에서 직접 실행해 얻었다. 재현 명령을 각 항목에 붙였다.
설계나 계획은 여기 없다 — 그것은 `DMTA_DESIGN.md`와 `IMPROVEMENT_PLAN.md`의 일이고,
이 문서는 **그 두 문서가 전제하는 코드의 현재 상태**를 잰다.

**한 문장 진단:**

> 이 저장소는 **자산을 소비하는 계층은 건강하고 테스트되어 있으며, 자산을 생산하는
> 계층은 테스트되지 않았고 일부는 이미 죽어 있다.** DMTA 루프는 전부 후자에 속한다.
> "정적으로 느껴진다"는 감각은 정확하며, 원인은 설계 미비가 아니라 **재생성 경로의 부재**다.

---

## 1. 정적인 이유 — 소비 계층과 생산 계층의 비대칭

저장소를 세 층으로 나누면 상태가 갈린다.

| 층 | 내용 | 상태 |
|---|---|---|
| (a) **커밋된 자산** | `assets/jak/*`, `assets/models/jak/*`, `assets/library/`, `assets/evidence/` | 완전. 오프라인 부팅 가능 |
| (b) **자산을 읽는 코드** | `funnel`, `conformal`, `applicability`, `deep_dive`, `app.py` | **176 테스트 통과 (115 s)** |
| (c) **자산을 만드는 코드** | `data/library`, `data/negatives`, `data/ingest`, `scripts/*audit*`, `reproduce.sh` | **부분적으로 죽어 있음 (아래)** |

(b)가 초록불이므로 저장소는 건강해 보인다. 그러나 **DMTA 루프에서 라운드마다 돌아야 하는
것은 전부 (c)** 다 — 재학습, 라벨 갱신, 라이브러리 재구축, 오라클 인구조사. (c)가
검증되지 않으면 루프는 원리적으로 돌 수 없다.

### 1a. 생산 경로 두 개가 이미 깨져 있다 (측정)

```
$ python -c "from src.data import library; library._gate_training_smiles()"
ImportError: cannot import name 'jak_positive_smiles' from 'src.data.negatives'

$ python -c "import scripts.gate0_audit"
ModuleNotFoundError: No module named 'src.data.jak'
```

- `src/data/library.py:76` — `negatives.jak_positive_smiles`를 import한다. 그 함수는
  STEP 15의 패널 일반화 때 `positive_smiles(panel=...)`로 개명됐고, 호출부는 따라가지
  않았다. **와이드 라이브러리는 재구축이 불가능하다.**
- `scripts/gate0_audit.py:26` — `src.data.jak`을 import한다. 그 모듈은
  `src/data/panel_data.py`가 됐다. **Gate 0 감사는 실행되지 않는다.**

두 경로 모두 `scripts/reproduce.sh`("저장소가 주장하는 모든 수치를 재현한다")에
들어 있다. 즉 **reproduce.sh는 지금 중간에서 죽는다.**

### 1b. 176개 테스트가 이것을 못 잡은 이유

우연이 아니라 구조다. `tests/test_library.py:26`이 **깨진 바로 그 심볼을
monkeypatch한다:**

```python
monkeypatch.setattr(lib, "_gate_training_smiles", lambda use_cache=True: set(excluded))
```

테스트가 대체하는 지점이 정확히 고장난 지점이다. 같은 패턴이
`tests/test_binder_gate.py:34`(`build_negatives` 대체),
`tests/test_loop_integration.py:45-50`(`build_isoform_dataset`, `load_library`, `_gate` 대체)에
있다. 이것은 테스트 위생 문제가 아니라 `DMTA_DESIGN.md` §2 B1이 지목한
**주입 seam 부재의 다른 얼굴**이다 — 데이터를 인자로 못 받으니 테스트는 모듈 전역을
갈아끼울 수밖에 없고, 그러면 진짜 생산 경로는 한 번도 실행되지 않는다.

> **결론:** 파이프라인이 정적인 1차 원인은 오라클 부재가 아니다. **자산을 다시 만드는
> 경로가 CI에서 한 번도 실행되지 않고, 이미 두 곳이 부러졌으며, 아무도 몰랐다는 것**이다.

---

## 2. 문서와 코드의 격차

최근 세 커밋(`c2847f3`, `b82e47a`, 그리고 그 사이 문서 갱신)은 **마크다운만** 추가했다.
두 설계 문서가 이름을 지어 둔 모듈의 현재 상태:

| 계획된 파일 | 상태 |
|---|---|
| `src/oracle.py`, `src/labels.py`, `src/dmta.py`, `src/acquire.py` | **전부 없음** |
| `scripts/dmta_run.py`, `scripts/oracle_audit.py`, `scripts/funnel_falsification_audit.py` | **전부 없음** |
| `src/docking.py` (README 모듈맵의 `future`) | 없음 |

`IMPROVEMENT_PLAN.md` §1의 핵심 수치(반증률 91.3 %)는 문서가 스스로 적었듯 **커밋된
스크립트로 재현되지 않는다.** 나는 이번 진단에서 그 수치를 재현하지 **않았다**. 대신 그
수치가 서 있는 모집단 계수는 독립적으로 재현했고 **정확히 일치한다**(§3c). 그래서 §1의
방법은 믿을 만하지만, **저장소에는 아직 그것을 다시 계산할 방법이 없다.**

Run History의 실사용도 문서가 적은 그대로다 — `append_round` 호출부 1개(`app.py:263`),
`rounds` 읽기 1개(`app.py:728`). 그 외 `registry`를 쓰는 코드는 없다.

---

## 3. 타깃 패밀리 일반화 — 명목상 되어 있고, 실질은 JAK 전용

이것이 기존 두 설계 문서가 다루지 않은 축이고, 사용자의 목표("JAK뿐 아니라 다른 타깃
패밀리")에 직접 걸린다. **`PanelSpec` 추상화 자체는 진짜다** — 경로, 캐시, 모델
네임스페이스가 전부 패널에서 파생되고 `PI3K` 패널이 등록되어 있다. 문제는 추상화가
아니라 **그 아래에 아무것도 없다는 것**이다.

### 3a. 자산 격차 — PI3K는 12개 중 5개만 있다

| 자산 | jak | pi3k |
|---|---|---|
| 아이소폼 데이터셋 parquet | 3 ✅ | 4 ✅ |
| `cross_measured.parquet` | ✅ (3 624행) | ✅ (1 500행) |
| `negatives.parquet` | ✅ | ❌ |
| `conformal_quantiles.json` | ✅ | ❌ |
| `gap_distribution.npz` | ✅ | ❌ |
| 회귀모델 `*_reg.pkl` | 3 ✅ | ❌ |
| `binder_gate.pkl` | ✅ | ❌ |
| AD 레퍼런스 `*.npz` | 3 ✅ | ❌ |

캠페인 카드는 이것을 정직하게 보고한다 (`campaign.build(PI3K)` → tier `bootstrap`,
`n_cross=1500`, `model_ids={}`). 앱은 "첫 실행 시 몇 분"이라고 안내한다.

### 3b. "첫 실행 시 몇 분"은 측정과 다르다

```
$ timeout 420 python -c "from src.models.binder_gate import train_and_cache; \
                         from src.panels import get_panel; train_and_cache(get_panel('pi3k'))"
Terminated                     # 7분에 미완, data/cache에 ChEMBL 활성 pull 11건 생성
```

이유는 `binder_gate` → `data.negatives.build_negatives` → **10개 음성 타깃을 ChEMBL에서
라이브 fetch**(타깃당 최대 4 000 레코드)이기 때문이다. 즉:

- 새 패널을 세우는 데 **네트워크가 필수**다. 오프라인 재현 경로가 없다.
- 배포된 Streamlit 앱(쓰기 불가, 임시 파일시스템)에서는 이 캐시가 남지 않는다 —
  `IMPROVEMENT_PLAN.md` §I가 "루프는 오프라인 스크립트"로 정한 것과 같은 제약이
  **패널 부트스트랩에도 걸린다.** 앱의 안내 문구는 이 사실을 반영하지 않는다.

### 3c. 라이브러리(건초더미)는 JAK 전용이다 — 측정

`library_molecule_overlap`을 두 패널에 돌렸다 (복원한 증거저장소, ChEMBL_37):

| 패널 | 라이브러리 38 592개 중 **해당 패널의 게이트 양성** |
|---|---:|
| jak | **8** (0.02 %) |
| **pi3k** | **134** (0.35 %) |

JAK의 8개는 `E2E_COMPLETION`이 보고한 잔여 누출과 일치한다(방법 교차검증). PI3K가
**16.8배**인 이유는 단순하다 — 라이브러리 구축 시 SMILES 배제는 **JAK 양성에 대해서만**
수행됐고(`library._gate_training_smiles`), 그 함수가 지금 깨진 것이다(§1a). 즉:

> **새 타깃 패밀리를 추가하려면 라이브러리를 그 패밀리의 활성을 배제하고 재구축해야
> 하는데, 재구축 경로가 정확히 그 지점에서 부러져 있다.** §1과 §3은 같은 결함의 두 얼굴이다.

### 3d. 오라클 설계가 패널 간에 이전되지 않는다 — 가장 중요한 신규 측정

`IMPROVEMENT_PLAN.md`는 `CensoredOracle`(라이브러리 안의 실측 비결합자)을 최우선
증거원으로 삼았다. 같은 질의를 두 패널에 돌렸다:

| 패널 | 아이소폼별 검열 측정 분자 | **라이브러리 ∩ 검열만 (합집합)** |
|---|---|---:|
| jak | 1 524 / 2 664 / 2 421 | **491** (문서의 491과 일치, 드러그라이크 414) |
| **pi3k** | 1 036 / 1 193 / 791 / 1 182 | **39** |

JAK 수치가 `DMTA_DESIGN.md` §1b와 **정확히 일치**하므로 방법은 맞다. 그리고 그 방법이
PI3K에서 내놓는 답은 **39개** — JAK의 8 %다. 드러그라이크 필터를 통과하면 더 줄어든다.

> **`DMTA_DESIGN.md` §1의 교훈이 한 패널 건너 그대로 반복된다.** "오라클 모집단은
> 설계 시점 가정이 아니라 런타임에 확인되어야 하는 수치"라는 규칙(G3)은 옳았고,
> 지금 그 규칙이 **PI3K에서 CensoredOracle을 비활성화시킬 가능성이 높다**는 것을
> 미리 측정으로 알 수 있다. 오라클 계층은 "구현하면 모든 패널에서 돈다"가 아니라
> **패널마다 인구조사를 먼저 해야 하는 구조**다.

### 3e. 그 외 JAK에 묶인 지점 (코드 위치)

- `PANELS` (`src/panels.py:127`) — 하드코딩된 dict 2개. 새 패밀리 = 코드 수정.
  UI/CLI에서 패널을 정의할 경로 없음.
- `VALIDATED_PANELS = {"jak"}` (`src/campaign.py:44`) — 손으로 유지하는 목록. 의도된
  설계이며 옳다(자동 승격 금지). 다만 **다른 패널이 승격되는 절차가 문서화되어 있지 않다.**
- `NEGATIVE_TARGETS` (`src/data/negatives.py:50`) — JAK 기준으로 고른 10개 바스켓.
  `usable_negative_targets`가 멤버 충돌만 제거한다. 새 패밀리에 대해 "하드 음성"인지는
  검사되지 않는다 (PI3K에는 PI3K 유사 리피드 키나아제가 하나도 없다).
- `app.py:381,542,640,796,809` — 기본값·문구·배지가 JAK 문자열.

---

## 4. DMTA 루프 — 기존 진단은 유효하다

`DMTA_DESIGN.md` §2의 B1–B6를 코드에서 재확인했고 **전부 현재 상태와 일치한다.**
중복 서술하지 않는다. 이 진단이 덧붙이는 것은 순서에 관한 하나다:

`IMPROVEMENT_PLAN.md` §5의 최종 순서는 `0 → A1 → F(주입 seam) → E(Run History)`로 시작한다.
그 순서 자체는 옳지만, **0순위 앞에 §1a의 두 줄짜리 수리가 들어가야 한다.** 이유:

- 0순위 `funnel_falsification_audit.py`는 라이브러리와 증거저장소를 함께 읽는다.
  라이브러리 재구축이 불가능한 상태에서 그 감사를 커밋하면, **감사 자체가 다시
  고정 자산에 묶인다** — 지금 진단하고 있는 바로 그 상태를 한 겹 더 쌓는 것이다.
- B(게이트 재설계)와 §3c(라이브러리 누출)는 같은 함수를 통과한다.

---

## 5. 우선 처리 권고 (진단에서 바로 따라 나오는 것만)

설계 결정을 요구하지 않고, 측정된 결함을 되돌리는 항목만 적는다.

| # | 항목 | 근거 | 크기 |
|---|---|---|---|
| 1 | `library.py:76`, `gate0_audit.py:26`의 죽은 import 수리 | §1a | 2줄 |
| 2 | `reproduce.sh` 또는 CI에 **생산 경로 스모크**를 넣어 §1a가 재발하면 실패하게 한다 | §1b | 작음 |
| 3 | 라이브러리 배제를 `panel` 인자화하고 PI3K 134개를 재측정 | §3c | 작음 |
| 4 | `scripts/oracle_audit.py`(계획된 0순위)를 **패널 인자로** 쓴다 — 단일 패널 스크립트로 만들지 않는다 | §3d | 설계 반영 |
| 5 | 새 패널 부트스트랩이 네트워크를 요구한다는 사실을 앱 문구와 README에 명시하거나, `negatives`를 증거저장소에서 읽도록 바꾼다 | §3b | 중간 |

1·2번이 들어가기 전까지 이 저장소에서 **"파이프라인이 돈다"는 (b)층에 대한 진술이지
파이프라인 전체에 대한 진술이 아니다.**

---

## 부록 — 재현

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q                                   # 176 passed
python -c "from src.data import library; library._gate_training_smiles()"   # ImportError
python -c "import scripts.gate0_audit"                                      # ModuleNotFoundError
python -m src.data.db --restore                       # assets/evidence -> 로컬 스토어
python -c "
from src.panels import PANELS, library_molecule_overlap
for n,p in PANELS.items(): print(n, library_molecule_overlap(p))"
```

검열 오라클 모집단(§3d)의 질의는 `assets/evidence`를 복원한 뒤
`activity.standard_relation='>' AND pchembl_value IS NULL`을 `library_member`와 조인하고,
같은 타깃에 pchembl 기록이 있는 분자를 제외한 것이다.
