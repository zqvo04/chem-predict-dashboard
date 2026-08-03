> **아카이브 (2026-08-03).** [ROADMAP.md](../../ROADMAP.md)로 대체됐다. §1의 오라클 인구조사는 STATE §5에, D0~D4 설계는 ROADMAP §4의 F·E·D·H 항목에 흡수됐다.
>
> **이 문서의 §1이 이 저장소의 방법론적 전환점이다** — README가 제안한 '공짜 프록시 오라클'이 9~28개뿐임을 세어서, 설계 전에 모집단을 세는 규범을 만들었다. ROADMAP §4의 '측정 먼저' 1단계가 그 규범의 직계 후손이다.

---

# DMTA 사이클 설계 — 오라클, Run History, 모델 피드백

**Status: design.** §1의 수치는 커밋된 증거 저장소(`assets/evidence/`, ChEMBL_37)와
커밋된 학습 자산(`assets/jak/`)에서 직접 측정했다. §2 이후는 계획이며, 결과가 나오면
VALIDATION.md로 간다.

이 문서의 자리는 기존 두 설계 문서 **사이**다.

| 문서 | 다루는 것 | 이 문서와의 관계 |
|---|---|---|
| `PHASE3_DESIGN.md` | 획득함수 벤치마크 (greedy/ucb/random 비교) | 회고적 오라클이 **이미 있다고 전제**한다. 그 전제를 만드는 것이 여기다. |
| `E2E_COMPLETION.md` | 라벨(P4.1)·게이트(P4.2)·누출(P4.3) 수정과 그 순서 | 순서 제약을 그대로 따른다. §7에서 어디에 끼우는지 명시. |
| **이 문서** | D-M-T-A를 실행 가능한 루프로 만드는 실행 계층 | 오라클 계약, Run History의 상태기계화, 모델 피드백 경로 |

**기록된 결정 두 가지** (프로젝트 오너에게 물어 답을 받음):
1. Test 단계의 1순위는 **증거저장소 프록시 오라클** — 배포된 루프가 실제로 새 정보를
   얻는 경로. (→ §1이 이 결정을 그대로 실행하지 못한다는 것을 측정으로 보인다.
   결정 자체는 유지되고, 구현 형태가 바뀐다.)
2. **인프라 먼저, 오라클은 P4.1 이후.** 라벨 정의와 무관한 부분(D0·D2)을 먼저 짓고,
   라벨에 의존하는 부분(D1·D3)은 assay-matched 라벨 수정 뒤에 붙인다.

---

## 1. 먼저 측정: 프록시 오라클은 제안된 형태로는 성립하지 않는다

README "Known gaps" §1은 프록시 오라클을 이렇게 제안했다 — *"채점된 분자를 전체
ChEMBL JAK 집합에서 조회해, 존재하지만 학습에 쓰이지 않았다면 실측 대 예측을 보고한다.
거의 공짜다."* 이 문장이 참인지가 오라클 설계 전체를 결정하므로, 먼저 셌다.

### 1a. 그 모집단은 비어 있다 (측정)

| 아이소폼 | 스토어에 pchembl이 있는 분자 | 그중 **학습셋에 없는** 분자 |
|---|---:|---:|
| JAK1 | 10 463 | **11** |
| JAK2 | 12 671 | **28** |
| JAK3 | 7 452 | **16** |
| 3종 모두 측정(=gap 답이 나오는 분자) | 3 621 | **9** |

오라클로 쓸 수 있는 분자가 **9~28개**다. 이건 오라클이 아니라 반올림 오차다.

이유는 사고가 아니라 **구조**다. 스토어는 학습셋을 만든 바로 그 타깃에서 인제스트됐고,
와이드 라이브러리는 JAK 활성을 배제하도록 만들어졌다. 즉 **건초더미와 오라클이 설계상
서로소**다. 퍼널이 채점하는 38 592개 라이브러리 분자 중 JAK pchembl을 가진 것은
JAK1 기준 5개뿐이고, 게이트 양성(any member pchembl ≥ 6)은 8개 — 이는
E2E_COMPLETION이 보고한 JAK 잔여 누출 8개와 정확히 일치한다(측정 방법의 교차 검증).

**"스토어를 조회하면 공짜 오라클이 나온다"는 것은 반증됐다.** 측정되지 않은 분자에는
측정값이 없다. 이건 인제스트를 더 해서 해결되는 문제가 아니다.

### 1b. 그러나 진짜 모집단이 두 개 있다 (측정)

**(A) 검열 레코드 — 측정된 비결합자.** `>` 10 µM로 측정되어 pchembl이 없는 기록.
학습 파이프라인이 pchembl NaN을 버리므로, **구성상 학습셋에 절대 들어가지 않는다.**

| 아이소폼 | 검열 측정 분자 | 그중 학습셋에 없음 |
|---|---:|---:|
| JAK1 | 1 524 | **1 281** |
| JAK2 | 2 664 | **2 039** |
| JAK3 | 2 421 | **2 258** |

**그리고 이것들 중 일부는 건초더미 안에 있다.** 와이드 라이브러리 38 592개 중
**491개(드러그라이크 414개)가 JAK 검열 기록만 보유**한다 — 누출이 아니라(활성이 아니므로
배제 필터에 걸리지 않았다), 배포된 퍼널이 실제로 채점하는 분자이면서 실측 증거가
붙어 있는 유일한 집합이다.

| 아이소폼 | 라이브러리 ∩ pchembl 보유 | 라이브러리 ∩ 검열 기록만 |
|---|---:|---:|
| JAK1 | 5 | **102** |
| JAK2 | 73 | **382** |
| JAK3 | 77 | **437** |

**(B) 시간 분할 — 진짜 gap 답이 나오는 유일한 경로.** 3종 교차측정 분자 3 621개
**전부가 최초 발표 연도를 갖는다**. 연도로 자르면 회고적 전향 시뮬레이션이 된다.

| 최초 연도 컷 | 컷 이후 분자 수 (= 오라클 모집단) |
|---|---:|
| ≥ 2015 | 3 180 |
| ≥ 2018 | 2 263 |
| **≥ 2020** | **1 525** |
| ≥ 2022 | 581 |

### 1c. 이 측정이 설계에 강제하는 것

세 가지가 따라 나오고, 아래 설계 전체가 여기서 나온다.

1. **오라클은 하나가 아니라 세 개**이며, 각각 **다른 질문에 답한다.** 하나로 뭉치면
   "루프가 정보를 얻었다"가 무엇을 뜻하는지 다시 불분명해진다.
2. **배포 루프가 실제로 얻을 수 있는 신호는 반증(falsification)뿐이다.** 건초더미 안의
   실측 증거는 전부 "이건 결합자가 아니다"이다. 이는 약점이 아니라 정확히 **바인더
   게이트와 potency floor가 하는 주장을 반증할 수 있는 신호**다.
3. **`>` 10 µM은 값이 아니라 상한이다.** 점 라벨로 쓰면 그 자체가 모델링 오류다
   (E2E_COMPLETION §6 Q2가 이미 지목). 평가에는 바로 쓸 수 있고(예측이 상한보다
   낮으면 맞은 것), 학습 라벨로 쓰려면 검열 회귀가 필요하다 — 이 둘을 분리해야 한다.

**즉시 얻어지는 부수 결과 하나.** 라이브러리 안의 414개 드러그라이크 검열 분자는
**바인더 게이트의 위양성률을 실제 건초더미에서 공짜로 측정할 수 있는 테스트셋**이다.
게이트는 지금까지 "부재로 추정된 비활성"에 대해서만 평가됐다. 이건 DMTA와 독립적으로
오늘 실행 가능하며, §7에서 0순위로 둔다. **경고: P4.2가 이 레코드들을 게이트 학습
음성으로 쓰려 하므로, 학습에 쓴 분자를 평가에도 쓰면 안 된다 — 분할을 먼저 못박아야
한다.** 이 충돌은 P4.2 설계가 아직 다루지 않았다.

---

## 2. 진단 — 지금 무엇이 루프를 막고 있는가

각 항목은 "무엇이 물리적으로 불가능한가"로 적는다.

### B1. 모델 계층이 학습 데이터를 인자로 받지 않는다 — 최상위 병목

`isoform_regressor.train_and_cache(panel, name, use_cache)`는 학습 데이터를
`panel_data.build_isoform_dataset` 캐시 경로에서 **스스로 읽는다**
(`src/models/isoform_regressor.py:120`). 동일한 결합이 `conformal.calibrate_gap`
(`src/conformal.py:121`), `conformal.halfwidth`(`:189`),
`applicability.load_reference`(`src/applicability.py:151`), `binder_gate`에도 있다.

→ **라운드마다 늘어난 라벨셋으로 재학습하는 것이 불가능하다.** PHASE3_DESIGN §3이
"per-isoform regressors, AD reference, conformal calibration을 매 라운드 refit한다"고
한 줄로 적은 것의 실제 비용이 이것이다. 지금 그렇게 하려면 모듈 전역을 monkeypatch
하는 수밖에 없고, 실제로 통합 테스트가 그렇게 한다
(`tests/test_loop_integration.py:45`). 테스트 편의를 위한 우회가 아니라, **주입 seam이
없다는 사실의 증상**이다.

예외: `applicability.build_reference(train)` / `in_domain(query, train=...)`은 이미
데이터를 인자로 받는다. AD 계층만 DMTA 준비가 되어 있다.

### B2. Run History가 쓰기 전용 로그다

`registry.append_round` 호출부는 저장소 전체에서 **한 곳**(`app.py:263`, kind="screen")
이고, `registry.rounds`를 읽는 곳도 **한 곳**(`app.py:728`, 화면에 표를 그린다)뿐이다.
`Round.kind`가 문서화한 `"select"` / `"rescore"`는 **한 번도 기록되지 않는다** —
컨트랙트를 내보내도 라운드가 남지 않는다.

`metrics`는 자유 dict이고(설계 의도였다), 그래서 **기계가 읽을 계약이 없다.**
라운드 N의 결과가 라운드 N+1의 선택을 바꾸지 못한다. 상태가 남지만 아무도 그 상태에
따라 행동하지 않는다 — **이건 루프가 아니라 감사 로그다.**

부수 결함(낮은 심각도, 기록만): `append_round`는 `index = len(rounds(...))`로 번호를
매긴 뒤 append한다(`src/registry.py:126`). 두 프로세스가 동시에 읽으면 같은 번호를
받고, `round_<n>_scores.parquet` 파일명이 충돌해 한쪽이 덮인다. 모듈 주석은 이 경우
"두 개의 라운드가 된다"고 하는데, 라운드 줄은 둘이지만 번호와 점수 파일은 하나다.
단일 사용자에서는 문제되지 않으나, CRO/협업자가 생기면 문제가 된다.

### B3. 라벨 원장이 없다

registry가 저장하는 것은 **점수표**(round_N_scores.parquet)이지 **측정값**이 아니다.
오라클이 값을 돌려줘도 쌓을 곳이 없고, 재학습이 읽을 대상이 없다. B1과 B3이 함께
있는 한, "모델 피드백"은 구현할 자리가 없다.

### B4. Test 단계 자체가 없다

README가 이미 정확히 지목한 격차다: Stage A는 유사체를 만들고 **그것을 제안한 바로
그 모델로** 재채점한다. gap S가 오르는 것은 모델이 자기 말에 동의한다는 뜻이다.
루프는 스키마로 닫히고(같은 컨트랙트, 같은 모듈, 동일성 assert) **증거로는 닫히지
않는다.**

### B5. 라운드별 재학습이 배포 모델을 파괴한다

모델 캐시 경로가 패널 단위다 — `panel.model_cache = data/models/<panel>`
(`src/panels.py:88`). 캠페인 루프가 라운드마다 재학습하면 배포된 JAK 회귀모델을
덮어쓰고, `current_model_ids`가 바뀌며, 이미 내보낸 **모든 컨트랙트의
`assert_models_match`가 깨지고**, `gap_distribution`의 provenance가 무효화된다
(`src/funnel.py:_gap_distribution_provenance`). PHASE3_DESIGN Gate E("JAK은
bit-identical")를 위반하는 경로가 여기다. 지금 설계에는 캠페인 사설 모델 네임스페이스가
없다.

다행히 `PanelSpec.root`가 필드다(`src/panels.py:47`, 테스트 때문에 그렇게 만들어졌다).
→ 파생 PanelSpec 하나로 해결 가능. §4 D2 참조.

### B6. Design 단계가 목적함수와 무관하다

`src/generate.py`는 방향족 위치에 고정 치환기 목록을 결정적으로 붙인다. 모델 예측도,
gap도, 이전 라운드 결과도 읽지 않는다. **라운드마다 같은 순서로 같은 분자를 낸다.**
SA 필터는 기본 임계 6.0에서 드러그라이크 시드에 대해 아무것도 거르지 않는다는 사실이
이미 테스트에 못박혀 있다. Design은 현재 사이클의 구성원이 아니라 상수다.

---

## 3. DMTA 현황 매핑

| 단계 | 있어야 할 것 | 현재 모듈 | 상태 |
|---|---|---|---|
| **Design** | 목적함수를 보고 후보를 제안 | `generate.py`(결정적 데코레이터), `funnel.screen_library`(카탈로그 선택) | 카탈로그 선택은 동작. 제안은 피드백 없음 (B6) |
| **Make** | 실현 가능성 판정 | `generate.sa_score`, `filters/druglikeness` | in-silico 게이트만. 구매가능성·역합성 없음 — **이 저장소에 합성은 없다. 그렇게 표기해야 한다.** |
| **Test** | 모델이 몰랐던 측정값 반환 | 없음 | **없음** (B4) |
| **Analyze** | 측정 대 예측을 채점하고 다음 라운드를 바꿈 | `registry`(로그만) | **없음** (B2·B3) |

Design–Make는 반쯤 있고, Test–Analyze는 없다. 그래서 이 설계의 무게는 T와 A에 있다.

---

## 4. 설계

### D0 — 학습 데이터 주입 seam (모든 것의 전제, 라벨 정의와 무관)

**변경 형태:** 새 추상화를 만들지 않는다. 기존 fit 함수에 선택적 인자를 하나 추가한다.

```python
def train_and_cache(panel, name, use_cache=True, data: pd.DataFrame | None = None)
def calibrate_gap(panel, alpha=..., seed=0, use_cache=True, cross: pd.DataFrame | None = None)
def load_reference(panel, isoform, use_cache=True, train: list[str] | None = None)
```

`None`이면 **현행 캐시 경로 그대로** — 배포 경로의 바이트가 변하지 않는다(Gate G0).
값이 주어지면 캐시를 읽지도 쓰지도 않고 그 데이터로 적합한다.

**왜 의존성 주입 클래스나 서브클래싱이 아닌가.** 호출부가 앱 하나와 스크립트 몇 개뿐이고,
필요한 능력은 "데이터를 바깥에서 준다" 하나다. 인터페이스를 새로 만들면 CLAUDE.md §2가
말하는 단일 사용처 추상화가 된다. 인자 하나가 최소이고, 부수 효과로
`test_loop_integration`의 monkeypatch가 정직한 인자 전달로 바뀐다.

**`LabelSet`** (`src/labels.py`, 작다): 라운드 시점의 라벨 상태 한 덩어리.

```
LabelSet
  per_isoform : dict[str, DataFrame]   # smi, pchembl, n_meas, censored, source
  cross       : DataFrame              # 교차측정 (gap 라벨)
  train_keys  : set[str]               # InChIKey — 오라클 누출 차단에 쓰인다
  provenance  : {source_id, round_index, cut, matching}   # P4.1의 매칭 기준을 담음
```

`train_keys`가 D1의 정직성 게이트를 가능하게 하므로, D0와 D1은 이 필드로 연결된다.

**게이트 G1:** 커밋된 데이터를 명시적으로 주입하면 캐시 경로와 **동일한 model_id**가
나온다. 두 경로가 같은 함수라는 증명.

---

### D1 — 오라클 계층 (P4.1 이후)

**공통 계약** (`src/oracle.py`):

```python
@dataclass(frozen=True)
class Measurement:
    smi: str; inchikey: str; isoform: str
    pchembl: float | None      # 검열이면 None
    relation: str              # '=' 또는 '>'
    bound: float | None        # 검열일 때의 상한 (pchembl 스케일)
    source_id: str; document_chembl_id: str | None; year: int | None

class Oracle(Protocol):
    name: str
    def query(self, smiles: list[str]) -> list[Measurement]: ...
    def population(self) -> int: ...      # 답할 수 있는 분자 수 — 설계 가정을 매번 재검증
    cost_per_query: float
```

`population()`을 계약에 넣는 이유는 §1 때문이다. **오라클이 답할 수 있는 분자 수는
설계 시점 가정이 아니라 런타임에 확인되어야 하는 수치**다. 9였다는 것을 문서를 읽어야
알게 되는 상황을 다시 만들지 않는다.

세 구현. 각각 답하는 질문이 다르므로 결과 표에서 절대 합치지 않는다.

| 구현 | 답하는 질문 | 모집단 (측정) | 답의 성격 |
|---|---|---|---|
| **`CensoredOracle`** | "퍼널이 유망하다고 한 분자가 실제로 비결합자인가?" | 라이브러리 내 414 (드러그라이크), 전체 1 281~2 258 | 반증. 상한 |
| **`TimeSplitOracle`** | "2020년 이전 데이터로 학습한 모델이 이후 화학을 찾아내는가?" | 1 525 (컷 2020) | gap 실측 |
| **`ManualOracle`** | "사람/CRO가 돌려준 값" | 0 (아직) | 값 또는 상한 |

**`CensoredOracle` — 배포 루프가 오늘 얻을 수 있는 유일한 실제 신호.**
증거 저장소에서 `standard_relation = '>' AND pchembl_value IS NULL`인 기록을 조회한다.
답은 "pchembl < 상한". 채점 규칙: 퍼널이 그 분자에 `meets_floor = True`(pred ≥ 6)를
줬는데 실측 상한이 5(=10 µM)라면 **모델이 틀렸음이 증명된다.** 이것이 지금 저장소에서
"모델이 자기 말에 동의" → "모델이 M번 중 N번 틀렸다"로 바꾸는 유일한 경로다.
바인더 게이트에 대해서도 같은 채점이 성립한다(위양성률).

**`TimeSplitOracle` — PHASE3의 회고 오라클을 대체한다.**
PHASE3_DESIGN §2는 pool 라벨을 **무작위로** 은닉했다. 연도 컷으로 은닉하는 편이 더
정직하다: 전향적 배치는 시간에 대해 무작위가 아니고, 무작위 은닉은 미래 화학형을
학습셋에 남겨 낙관 편향을 만든다. 모집단이 충분하다는 것은 §1b에서 측정됐다
(2020 컷 1 525개). 비용은 학습셋이 3 621 → 2 096으로 줄어드는 것이며, 이는 **정직해진
값이지 손실이 아니다.** PHASE3_DESIGN의 획득함수 비교는 이 오라클 위에서 그대로 돈다.

**정직성 규칙 (구현 전에 못박을 것):**
1. **누출 차단.** `LabelSet.train_keys`에 있는 InChIKey는 오라클이 **반드시 None을
   반환**한다. 학습에 쓴 분자를 조회한 것은 오라클 답변이 아니라 회상이다. 테스트로
   고정(G2).
2. **검열은 기본적으로 평가 전용.** 학습 라벨로 승격하려면 검열 회귀를 쓰거나,
   "상한을 점 라벨로 취급한다"는 결정을 명시적으로 기록한다. 조용한 승격 금지.
3. **프록시 오라클의 커버리지 수치는 보증이 아니다.** 문헌에 측정된 분자는 풀의 균일
   표본이 아니다(누군가 시험할 이유가 있었던 분자다). PHASE3 §5의 랜덤 스트림이 주는
   교환가능성은 `TimeSplitOracle`에서도 성립하지 않는다 — 시간 분할은 의도적으로
   교환가능성을 깬다. **따라서 오라클 위에서 측정한 커버리지는 "지지 증거"로만 보고하고,
   STEP 14의 보증과 같은 칸에 쓰지 않는다.**

---

### D2 — Run History를 상태기계로 (라벨 정의와 무관, 지금 착수 가능)

세 가지를 추가한다. 기존 JSONL 구조는 유지한다(그 판단은 옳았다).

**(a) kind별 타입 있는 metrics.** `metrics`는 자유 dict로 두되, kind마다 dataclass를
정의하고 기록 시 직렬화·읽을 때 검증한다. 자유 dict가 문제가 아니라 **읽는 쪽이
무엇을 기대하는지 아무도 적지 않은 것**이 문제였다.

```
DesignMetrics   n_proposed, source(screen|generate), seed, acquisition, kappa
TestMetrics     oracle, n_queried, n_answered, n_censored, hit_rate,
                mae_on_answered, interval_coverage
AnalyzeMetrics  retrained(bool), model_ids_before/after, delta_spearman, trigger
```

새 kind: `design`, `test`, `analyze`. 기존 `screen`/`select`/`rescore`는 유지하고,
**`select`를 실제로 기록한다**(현재 미기록, B2).

**(b) 라벨 원장 `labels.parquet`** — 캠페인 디렉토리에 append-only.

```
smi, inchikey, isoform, pchembl, relation, bound,
revealed_round, oracle, source_id, document_chembl_id, year
```

오라클이 밝힌 모든 측정이 여기 쌓이고, **재학습은 오직 여기서만 읽는다.** 이것이 B3의
해소이며, 재개 가능성의 근거다. append-only인 이유는 registry와 같다: 측정은 사실이고
사실은 수정되지 않는다(재측정은 새 행이다).

**(c) 읽기 API — 루프가 상태를 묻는 곳.**

```python
def state(campaign_id) -> CampaignState   # next_round, labelset, last_model_ids, spent_budget
```

이것이 없으면 재개(PHASE3 Gate A)가 성립하지 않는다. **재개가 동일해지려면 seed와
획득 파라미터가 라운드에 기록되어 있어야 한다** — 그래서 (a)의 `seed`가 선택 필드가
아니다.

**(d) 캠페인 사설 모델 네임스페이스 — B5 해소.**

```python
campaign_panel = replace(panel, root=campaign_dir(cid) / f"round_{n}")
```

`PanelSpec.root`가 이미 필드이므로 새 코드가 거의 필요 없다. 라운드 모델은
`data/registry/<cid>/round_<n>/models/`로 가고, **`assets/models/jak/`는 절대 건드리지
않는다.** Gate G0가 이를 고정한다.

---

### D3 — 모델 피드백: 세 채널을 분리한다

"모델 피드백"이라는 한 단어에 성격이 다른 세 가지가 들어 있고, 섞으면 무엇이 개선됐는지
말할 수 없게 된다.

**채널 1 — 라벨 피드백 (재학습).**
`labels.parquet` + 기존 학습셋 → 새 `LabelSet` → D0의 주입 경로로 재적합.
*재학습 트리거 정책 (권장):*
- 회고 벤치마크: **매 라운드 재학습.** 프로토콜이 그렇게 정의되어 있고 비교 가능성이
  중요하다.
- 배포 캠페인: **임계 기반 + 사람의 명시적 승인.** 재학습은 `model_id`를 바꾸고 이미
  내보낸 컨트랙트를 무효화한다. 조용히 일어나면 안 된다. 트리거 후보: 신규 라벨 N개
  누적, 또는 오라클 채점 오차가 보고된 검증 오차를 초과.

**채널 2 — 캘리브레이션 피드백.**
PHASE3_DESIGN §5(랜덤 스트림)를 그대로 채택하되, D1 정직성 규칙 3의 단서를 붙인다.
프록시 오라클에서 얻은 커버리지는 교환가능성이 없으므로 **모니터링 지표**이지 보증이
아니다. 이 구분을 UI 문구까지 끌고 간다.

**채널 3 — 설계 피드백 (가장 작게, 그리고 과장 없이).**
최소안: 라운드 N에서 **어떤 치환기가 gap을 올렸는지 통계를 기록**하고, 라운드 N+1의
후보 생성 순서를 그 통계로 재정렬한다. `generate.SUBSTITUENTS`가 고정 리스트이므로
변경은 작다.
**과설계 경고를 문서에 남긴다:** 이것은 생성 모델이 아니라 **가중 열거**다. "생성형
설계"로 표기하지 않는다. 진짜 생성 모델은 Colab/GPU seam의 일이고, 그 seam은 이미
문서화되어 있다.

---

### D4 — 오케스트레이션 (`src/dmta.py`)

한 라운드 = 네 단계, 각 단계가 라운드를 기록한다.

```
run_round(campaign_id, oracle, acquisition, budget):
    state   = registry.state(campaign_id)              # 재개 지점
    DESIGN  : 후보 = funnel.screen_library(...) 또는 generate.analogues(...)
              획득함수로 batch 선택  → append_round("design", DesignMetrics)
    MAKE    : druglikeness + SA + (미래) 구매가능성    → design 라운드에 필드로 기록
    TEST    : oracle.query(batch)  (train_keys 차단)   → append_round("test", TestMetrics)
              labels.parquet 에 append
    ANALYZE : 예측 대 실측 채점, 트리거 판정, 필요시 재학습
                                                       → append_round("analyze", AnalyzeMetrics)
```

**Make에 대한 정직한 표기.** 이 저장소에 합성은 없다. Make는 in-silico 실현가능성
게이트이며, 현재는 SA 점수와 드러그라이크니스뿐이다. 문서·UI 어디서도 Make를
"합성"이라 부르지 않는다. 구매가능성(ZINC/Enamine 조회)은 값싼 다음 항목이지만
이 설계의 범위 밖이다.

---

## 5. 파일 계획

| 파일 | 책임 | 신규/변경 |
|---|---|---|
| `src/labels.py` | `LabelSet` — 라운드 시점 라벨 상태 + train_keys | 신규 (작음) |
| `src/oracle.py` | `Oracle` 프로토콜 + `CensoredOracle` / `TimeSplitOracle` / `ManualOracle` | 신규 |
| `src/dmta.py` | 라운드 오케스트레이션, 재개 | 신규 |
| `src/acquire.py` | 획득함수 (순수 함수) | 신규 — PHASE3_DESIGN §7 그대로 |
| `src/registry.py` | typed metrics, labels 원장, `state()`, 새 kind | 변경 |
| `src/models/isoform_regressor.py`, `src/conformal.py`, `src/applicability.py`, `src/models/binder_gate.py` | 선택적 `data=` 인자 | 변경 (서명만) |
| `scripts/dmta_run.py` | 캠페인 한 개를 N라운드 돌린다 | 신규 |
| `scripts/oracle_audit.py` | 각 오라클의 `population()` 보고 — §1 표를 재생성 | 신규 |
| `app.py` | Run History 탭: 라운드 타임라인 + 오라클 채점 성적 | 변경 |

---

## 6. 게이트 — 무엇을 만족해야 "됐다"인가

| # | 게이트 | 왜 |
|---|---|---|
| **G0** | JAK 배포 자산 불변: `model_id`, `gap_distribution` provenance, 컨트랙트 검증이 전부 동일 | DMTA는 위에 얹는 것이지 배포된 스크린을 흔드는 것이 아니다 (B5, PHASE3 Gate E) |
| **G1** | 커밋 데이터를 명시적으로 주입한 결과 == 캐시 경로 결과 (동일 model_id) | D0의 seam이 같은 함수임의 증명 |
| **G2** | `train_keys`에 있는 분자를 조회하면 오라클이 **반드시 None** | 누출 차단. 이게 없으면 모든 성적이 회상이다 |
| **G3** | 각 오라클의 `population()`이 보고되고, 0에 가까우면 그 오라클은 **비활성화된다** | §1의 실패를 반복하지 않는다 |
| **G4** | 라운드 3에서 중단 후 재개 시 라운드 4가 동일 | 스크립트가 아니라 루프임의 정의 (PHASE3 Gate A) |
| **G5** | 재학습 후 오차 개선 **또는 미개선이 동일한 비중으로 보고** | 부정 결과도 결과다. 이 저장소의 기존 규범 |
| **G6** | 게이트 위양성률이 라이브러리 내 414개 실측 비결합자에서 보고됨, 학습/평가 분할 명시 | §1c의 부수 결과. P4.2와 충돌하지 않게 |

**무엇이 "DMTA가 작동한다"를 반증하는가** (미리 적어 둔다):
오라클이 답한 분자에서 모델의 오차가 **학습셋 밖 홀드아웃 오차와 구분되지 않고**,
재학습 후에도 그 오차가 seed 간 산포 안에서 움직인다면, 결론은 **"이 규모에서 루프는
정보를 얻지 못한다"**이며 그대로 VALIDATION.md에 간다. 1 525개(시간 분할)와 414개
(검열)는 큰 효과는 볼 수 있지만 작은 효과는 볼 수 없는 크기다 — 이 한계를 결과와 함께
적는다.

---

## 7. 순서

E2E_COMPLETION §2의 순서 제약(라벨 → 게이트 → 건초더미 → AL)을 지키면서, 라벨 정의와
직교한 작업을 앞으로 당긴다.

```
  0    오라클 인구조사 + 게이트 위양성 측정   (측정만, 반나절)   ← §1을 스크립트로 고정
  D0   학습데이터 주입 seam                  (라벨과 직교)
  D2   Run History 상태기계화 + 라벨 원장     (라벨과 직교)
  ──────────  여기까지는 P4.1과 병행 가능  ──────────
  P4.1 assay-matched 라벨                    (E2E_COMPLETION)
  P4.2 측정 음성 → 게이트   / P4.3 누출 수정
  D1   오라클 구현                            (P4.1의 매칭 기준을 답에 반영해야 함)
  D3   모델 피드백 / 재학습 트리거
  D4   오케스트레이션 + 재개
  ──────────
  Phase 3  획득함수 벤치마크 (TimeSplitOracle 위에서)
```

**D1이 P4.1 뒤인 이유**는 E2E_COMPLETION의 논리 그대로다. 오라클이 돌려주는 gap이
same-document 매칭으로 정의될 것이라면, 그 이전에 만든 오라클은 곧 폐기될 정의로
채점하게 된다. 반면 **D0와 D2는 라벨이 무엇으로 정의되든 동일**하므로 기다릴 이유가
없고, P4.1 자체도 D0의 주입 seam이 있으면 더 쉽게 검증된다(매칭 기준별 학습셋을
인자로 넣어 비교).

**0순위를 측정으로 시작하는 이유**는 이 문서 자체가 증명한다. §1을 재기 전의 설계는
존재하지 않는 모집단 위에 세워질 뻔했다.

---

## 8. 리스크와 미해결 질문

| 리스크 | 왜 그럴듯한가 | 완화 |
|---|---|---|
| **오라클 모집단이 또 비어 있다** | §1에서 실제로 일어났다 | G3: `population()`을 계약에 포함, 0이면 비활성화 |
| **검열 상한을 점 라벨로 써서 모델을 망친다** | 편해서 그렇게 하게 된다 | D1 규칙 2: 기본 평가 전용, 승격은 명시 결정 |
| **P4.2가 쓴 음성으로 P4.2를 평가한다** | 같은 7 119개 분자가 학습·평가 양쪽 후보 | G6: 분할을 P4.2 착수 전에 못박는다 |
| **재학습이 배포 컨트랙트를 조용히 깬다** | 캐시 경로가 패널 단위 | D2(d) 사설 네임스페이스 + G0 |
| **커버리지 수치가 보증처럼 읽힌다** | 같은 단어, 다른 의미 | D1 규칙 3: 표와 UI에서 칸을 분리 |
| **채널 3이 "생성형 AI"로 과대 표기된다** | 유혹이 크다 | 문서에 "가중 열거"라고 못박음 |

**미해결 질문 (결정 필요):**

1. **시간 컷을 어디로 둘 것인가.** 2020(오라클 1 525 / 학습 2 096)이 기본 권장이나,
   2018(2 263 / 1 358)은 오라클이 크고 모델이 약해진다. 두 컷 모두 돌려 보고할지,
   하나로 못박을지.
2. **`CensoredOracle`의 상한 값 처리.** ChEMBL의 `>` 기록은 상한이 10 µM만이 아니다
   (다양하다). pchembl 스케일로 환산해 그대로 쓸지, 단일 임계로 이산화할지 — 전자가
   정직하고 후자가 단순하다.
3. **라벨 원장을 registry(JSONL/parquet)에 둘 것인가, 증거 저장소(DuckDB)에 둘 것인가.**
   현재 설계는 registry에 뒀다(캠페인 지역 상태이므로). 다만 CRO/협업자가 생기면
   E2E_COMPLETION §4의 Postgres 트리거가 바로 여기서 당겨진다.
4. **PHASE3의 랜덤 은닉을 시간 은닉으로 바꾸는 것을 확정할지.** 이 문서는 그렇게
   권장하지만, PHASE3_DESIGN의 기존 수치(랜덤 시드 5개 프로토콜)와의 비교 가능성을
   포기하게 된다.
