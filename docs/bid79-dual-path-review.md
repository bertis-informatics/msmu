# BID-79 재검토 — "이중 경로 제거" 관점

리뷰 질문을 바꿔서 봤습니다. 기존 검토는 *"지금 두 경로가 다 맞는가"* 였고,
이번은 **"legacy 경로(pandas / dense)를 지웠을 때 무엇이 사라지는가"** 입니다.

전부 `.venv`(polars 1.43.0 / pandas 3.0.0 / anndata 0.12.0 / scipy 1.17.0)에서 실행해 재현.
`CONFIRMED` = 실행 재현, `ARGUED` = 코드 라인 논증.

---

## 0. 핵심 결론

**이미 고친 MaxQuant `cast(Utf8)` 버그는 단발 사고가 아니라 5건짜리 계열이었습니다.**

기전이 전부 동일합니다:

```
polars가 pandas와 달리 null을 반환
  → to_pandas()가 컬럼을 float64로 승격
  → _base_reader.py:310 의 astype(str)이 "2243" 대신 "2243.0" 을 모든 행에 렌더
  → identification 인덱스와 quantification 인덱스의 교집합이 사라짐
  → 피처가 조용히 증발 (최악의 경우 모달리티 전체가 빈 채로 반환)
```

한 행의 불량이 파일 전체를 날리는데 예외가 안 납니다.
**pandas 경로가 지금 이걸 전부 가려주고 있고, 삭제와 함께 노출됩니다.**

| 영역 | 상태 |
|---|---|
| **polars null 처리** | **위험 — 계열 결함 5건** (§A). 최우선 |
| **polars 파서 기본값** | **위험** — `pl.read_csv` 기본 인자가 검색엔진 출력에 안전하지 않음 (§B) |
| **sparse 경계 조건** | **위험** — `split_tmt`이 sparse 입력에서 결측을 0으로 오염, 양쪽 분기 모두 (§C) |
| polars 변환 커널 | 대체로 깨끗 — 오히려 pandas보다 **정확한 곳이 4군데** (§D) |
| sparse 수치 커널 | 깨끗 — 랜덤 900회 0 불일치, h5mu 왕복 패턴 보존 (§F) |
| **테스트 커버리지** | **위험** — parity 테스트가 곧 oracle. 지우면 검증 능력도 같이 죽음 (§E) |

**지금 당장**: §A-1 (Sage), §A-2 (Sage), §A-3 (DELPI), §C-1 (`split_tmt`).

---

## A. polars null 처리 — 조용히 데이터가 사라지는 계열 5건

### A-1. `_sage.py:133` — scannr 파싱 실패가 **모달리티 전체를 비움** · CONFIRMED · **최고 심각도**

```python
_sage.py:133  (polars)  scan= 뒤 숫자 추출, 실패 시 null
_sage.py:99 / :79 (pandas)  int(...) → 실패 시 IndexError / ValueError (시끄럽게 실패)
```

트리거: `scan=<digits>` 에 안 맞는 모든 `scannr` —
`"12345"`, `"MERGED_SCANS_2000_2010"`, `"scan= 2243"`(공백), `"scan=-5"`, `"scan=2243 index=1"`.

증폭 결과 실측:
```
ident 인덱스 : ['RunA.2243.0', nan, 'RunB.7.0']     <- 한 행이 null → 컬럼 전체 float64
TMT quant    : ['RunA.2243', 'RunB.7']
intersection : []                                   <- 모달리티가 빈 채로 반환
```
**불량 1행이 파일 전체를 침묵으로 파괴합니다.**

### A-2. `_sage.py:202` — TMT quant 인덱스가 선행 0을 안 벗김 · CONFIRMED

`scannr = "scan=002243"` 일 때 pandas quant는 `int()` 를 거쳐 `RunA.2243`,
polars는 `RunA.002243`. ident 쪽(`:133`)은 캐스팅을 하므로 양쪽 다 `RunA.2243` →
**ident∩quant 가 3행 → 1행**.

FragPipe 형제(`_fragpipe.py:226`)에는 `.cast(pl.Int64).cast(pl.Utf8)` 가 바로 이 케이스
주석과 함께 들어 있습니다. **Sage 쪽만 누락** — 단순 실수로 보입니다. 1줄 수정.

### A-3. `_delpi.py:86` — 고친 MaxQuant 버그가 DELPI에 그대로 · CONFIRMED

```python
_delpi.py:86  pl.col("run_name").cast(pl.Utf8) + "." + pl.col("pmsm_index").cast(pl.Utf8)
_delpi.py:59  df["filename"].astype(str) + "." + df["pmsm_index"].astype(str)   # to_pandas 이후
```
직접 재현:
```
polars cast(Utf8) : ['r.1',    None,   'r.3']
astype(str) 경로  : ['r.1.0', 'r.nan', 'r.3.0']      <- 교집합 0
```
`read_delpi()` 기본 경로: `pandas psm (1,5)` vs `polars psm (1,1)` — **5개 중 4개 증발**.
`tests/` 에 DELPI 테스트가 없어 아무것도 못 잡습니다.
수정은 MaxQuant와 동일: `select` → `to_pandas()` → `self._make_unique_index(...)`.

### A-4. `_sage.py:142` — null `label` 이 target/decoy **양쪽에서** 행을 지움 · CONFIRMED

pandas `NaN == -1` → `False` → `0`(int64). polars `null == -1` → `null` → float64 NaN.
`_base_reader.py:452, :486-487` 이 `== 0` / `== 1` 로 거르는데 NaN은 어느 쪽에도 안 걸립니다.
```
입력 3행 → pandas: target 2 + decoy 1 = 3
          → polars: target 1 + decoy 1 = 2
```
수정: `(pl.col("label") == -1).fill_null(False).cast(pl.Int64)`.

### A-5. `_fragpipe.py:171-172` / `_maxquant.py:23` · CONFIRMED

- **FragPipe**: null `Spectrum` → A-1과 동일 기전 (`filename=NaN, scan_num=NaN`, 전 행 `.0` 오염).
- **MaxQuant**: `~pl.col("Type").is_in([...])` 에서 `null.is_in()` → `null` → `~null` → `null` →
  `filter` 가 버림. pandas는 `NaN.isin()` → `False` → `~` → `True` → 유지.
  ```
  pandas keeps: [101, 102, 104]     <- Type null 인 102 유지
  polars keeps: [101, 104]          <- 102 소실
  ```
  **모든 MaxQuant 읽기의 기본 경로**입니다.

### A-6. 유령 그룹 — null 키가 진짜 그룹으로 승격 · CONFIRMED

`_maxquant.py:214-217`: pandas `pivot_table` 은 NaN 그룹 키를 버리고, polars `group_by`+`pivot` 은
유지 → null `Raw file` 이 있으면 `obs_names = ['null', 'raw_0', 'raw_1']`,
**문자 그대로 `"null"` 이라는 가짜 샘플**. `_delpi.py:91` 도 동형.
sparse 쪽에도 같은 패턴(§C-2의 `nan` 유령 plex)이 있습니다.

---

## B. polars 파서 기본값

`_base_reader.py:53-63` 이 `infer_schema_length` / `null_values` / `schema_overrides` 를 전부 기본값으로 둡니다.

| # | 항목 | pandas | polars (현재) |
|---|---|---|---|
| B-1 | 150행 정수 뒤 `123.456` | `float64` | **`ComputeError`** (앞 100행만 보고 `i64` 확정) |
| B-1 | 120행 정수 뒤 `unknown` | `object` | **`ComputeError`** |
| B-2 | `NA`/`null`/`None`/`#N/A` | `float64`+NaN | **`String`** (리터럴 유지, 조용함) |
| B-2 | `NaN` | `float64`+NaN | `Float64`+NaN ✔ |
| B-3 | 전부 빈 컬럼 | `float64` | **`String`** |

**B-2가 Sage에서 조용히 통과합니다** — `rt, expmass, calcmass, charge, peptide_len,
missed_cleavages, semi_enzymatic, hyperscore, spectrum_q` 가 전부 캐스팅 없는 `pl.col(...)`
carry-through라 **문자열이 그대로 `var` 에 실립니다**. (FragPipe는 `_fragpipe.py:175` 에서
`InvalidOperationError` 로 시끄럽게 죽음 — 그나마 나음.)

**B-3은 end-to-end로 터집니다** — MBR 미사용 시 `Lib.Q.Value` 가 비면 `_diann.py:182` 에서
`InvalidOperationError: sum operation not supported for dtype str`. pandas는 정상.

### B-4. 다중 파일 concat · CONFIRMED

`_base_reader.py:206` `pl.concat(how="vertical_relaxed")` 는 **dtype만** 완화, 스키마는 동일해야 함:
- 컬럼 집합 다름 → `ComputeError: schema lengths differ`
- **컬럼 순서만 달라도** → `ComputeError: schema names differ` (pandas는 순서 무관)

현실 케이스: **DIA-NN `Decoy` 유무 혼합**이 진짜 blocker입니다 — 코드 자체가
`_diann.py:221` 에서 `Decoy` 를 optional로 취급하므로 **의도된 입력**인데 polars가 거부합니다.
(TMT10+TMT16 혼합도 걸리지만, `_maxquant.py:180` 이 컬럼 수로 plex를 유도하므로 어차피 의미상 잘못된 입력 → 허용 가능한 손실.)

### B-5. 검증된 수정안

| 수정 | 닫히는 항목 |
|---|---|
| `infer_schema_length=None` | B-1 전부 |
| `null_values=<pandas STR_NA_VALUES>` | B-2 전부 |
| `how="diagonal_relaxed"` | B-4 전부 (dtype 완화 유지한 채 컬럼 union) |
| — | **B-3은 안 닫힘** — 추론할 데이터가 없음. `schema_overrides` 또는 read 후 cast 필요 |

`infer_schema_length=None` 은 전체 스캔 비용. 리더가 컬럼명을 이미 다 알고 있으니
`schema_overrides` 로 숫자 컬럼만 못박는 편이 낫습니다.

> **한계**: 레포 픽스처(`tests/*.tsv`)는 작고 깨끗해서 B-1/B-2 **재현 안 됨**.
> 메커니즘은 확정, 실데이터 발생 여부 미확인. 보유한 실제 report에 돌리면 즉시 판정:
> ```bash
> python -c "import polars as pl,pandas as pd,sys;f=sys.argv[1];a=pd.read_csv(f,sep='\t');b=pl.read_csv(f,separator='\t');print([(c,str(a[c].dtype),str(b[c].dtype)) for c in a.columns if str(a[c].dtype).startswith(('int','float'))!=str(b[c].dtype).startswith(('Int','Float'))])" report.tsv
> ```

### B-6. `pd.DataFrame` 리스트 입력은 **이미 오늘 깨져 있음** · CONFIRMED

`_read_file` 이 `_base_reader.py:157-158` 에서 `as_polars` 검사 **전에** short-circuit 하므로
DataFrame은 항상 pandas로 나옵니다. 그런데 2개 이상이면 `_base_reader.py:206` `pl.concat` 이
`TypeError: did not expect type: 'pandas.DataFrame' in 'concat'`.
혼합 `[path, DataFrame]` 은 `AttributeError: ... has no attribute 'lazy'`.
**기본 경로의 현존 버그**입니다. `read_*` 진입점들이 `isinstance(x, list)` 만 검사하고
원소 타입은 안 봅니다 (`_reader_registry.py:43-48` 외 4곳).

기타 pandas로 폴백되는 입력: `.xlsx`/`.xls`, `.json`. URL은 폴백 아님 —
`pl.read_csv(url)` 은 되지만 **`pl.read_parquet(url)` 은 HTTP Range 지원 필요**
(없으면 `OSError`). `tests/test_tutorial_dia-lfq.ipynb` 가 GitHub raw에서 `.parquet` 을
읽고 pytest가 이걸 실행합니다(`pyproject.toml:73` `--nbval-lax`).

---

## C. sparse 경계 조건

### C-1. `split_tmt` 이 sparse 입력에서 결측을 0으로 오염 — **양쪽 분기 모두** · CONFIRMED · **최우선**

```python
_tmt.py:91   source = source_x.toarray() if sp.issparse(source_x) else ...   # sparse 분기
_tmt.py:115  df = psm_adata.to_df().T.copy()                                 # dense 분기 (기본값!)
```
`toarray()` / `to_df()` 가 구조적 결측을 0으로 채우고 아래에서 `np.isfinite(0) == True` 라
**결측이 "관측된 0"으로 승격**됩니다. `_blockdiag.py` 가 존재하는 이유인 바로 그 invariant 위반.

재현 (입력 8셀 중 1개 결측):
```
input nnz 7 → output nnz 8          <- 결측이 저장된 0으로
dense_block[2,1] = 0.0                (dense 정답: nan)
한 채널 nanmedian 11.0 → 5.5, 이후 log2_transform 이 저장된 -inf 생성
```
**도달 경로**: `read_diann(sparse=True)` → `split_tmt(...)`, 또는 `split_tmt` 2회 호출.
sparse-only 최종 상태에서는 입력이 항상 sparse이므로 **모든 TMT split이 걸립니다.**
`sparse=False` 가 기본값이라 **기본 분기가 더 잘 걸립니다.**
수정: `dense_block(source_x)` (더 낫게는 COO row index 재매핑으로 densify 회피).

### C-2. map 누락이 `nan` 유령 plex 생성 · CONFIRMED

```
sparse=False: KeyError: "['p1','p3'] not in index"           <- 시끄럽게 실패 (올바름)
sparse=True : 정상 종료, obs = [..., 'c126_nan', 'c127_nan']  <- 가짜 plex
```
dense가 제공하던 유일한 방어선이 삭제와 함께 사라집니다.

### C-3. sparse에서 아예 죽는 기능 2개 · CONFIRMED

| 위치 | 증상 |
|---|---|
| `_summarisation.py:695` `PtmSummarisationPrep.prep` | `TypeError: Can only merge Series or DataFrame objects, a <class 'SparseQuant'>` → **`to_ptm` 사용 불가**. base `prep()` 의 densify 가드가 이 서브클래스엔 없음 |
| `_summarisation.py:511-526` | `median_polish` / `directlfq` → `NotImplementedError` → **`to_protein(agg_method="median_polish")` 사용 불가** |

`median_polish` 는 rollup 연구 결론의 핵심이라 sparse-only 전환의 **실질적 blocker**.

### C-4. `_utils/_anndata.py:32` `_has_quant_values` 가 sparse에서 반전 · CONFIRMED

`if sparse_matrix.nnz < rows*cols: return True` — 전부 결측인 sparse(`nnz==0`)가
`has_quant=True` 를 반환 (동일한 all-NaN dense는 `False`).
`_summarise.py:157` 분기가 뒤집혀 LFQ/DDA에서 `to_peptide` 가 기존 peptide modality 대신
all-NaN 행렬을 새로 만듭니다. `nnz < rows*cols` 는 "sparse인가"이지 "값이 있는가"가 아님.

### C-5. `SparseQuantFrame` pass-through 가 리더 훅을 건너뜀 · ARGUED

`_base_reader.py:403` 이 `_make_needed_columns_for_quantification` /
`_make_rename_dict_for_obs` / `replace(0,nan)` 을 전부 건너뜁니다.
DIA-NN은 둘 다 오버라이드하지 않아 **지금은 정확**하지만
**Sage / MaxQuant / FragPipe 는 전부 오버라이드**합니다 (grep 확인).
sparse를 저 리더로 확장하는 순간 obs 이름 변경과 quant 컬럼 선택이 조용히 건너뛰어집니다.

---

## D. polars가 **더 정확한** 곳 — 삭제가 오히려 버그를 고치는 4건

균형을 위해: 전부 pandas 쪽이 틀렸고 polars가 맞습니다.

1. **`_sage.py:196`** — pandas는 drop되지 않은 모든 컬럼을 유지, polars는 `tmt_*` 만.
   `_make_rename_dict_for_obs`(`:212`)가 `plex = len(columns)` 로 세므로 여분 컬럼 하나가
   3-plex에서 `Tmt4` 조회 → `AttributeError`. **polars가 정답.** (현재 픽스처엔 여분 컬럼이 없어 잠복.)
2. **`_sage.py:181-183`** — pandas 분기가 `search_settings.quantification_df` 를 **제자리 변형**
   (`scan_num`, `tmp_index` 추가). 두 번째 `read()` 에서 plex 오산. polars는 안 건드림.
3. **`_fragpipe.py:165-168`** — 리터럴 `nan` 토큰 필터가 pandas는 strip 전, polars는 strip 후.
   polars 결과가 맞음.
4. **CSV float 파싱 1-ULP** — 실제 `results.sage.tsv` 에서 `hyperscore` 5000개 중 68개가
   3.55e-15 차이. 원문 `27.437794026619954` 를 polars는 정확히, pandas는 `...61995` 로 파싱.
   **polars가 correctly-rounded.** ← 기존 골든 `.h5mu` 대조 테스트가 이걸 감지할 겁니다.

추가로 **FragPipe pandas 분기는 이 venv의 pandas 3.0에서 아예 못 돕니다**
(`_fragpipe.py:123` `AttributeError`, `:130` `TypeError`) — 포팅과 무관한 선재 결함.
즉 **FragPipe는 pandas를 oracle로 쓸 수 없습니다.**

또한 **행 순서**: pandas 경로는 `as_completed`(`_base_reader.py:217-225`) 때문에
**비결정적**(3회 시행에서 첫 행이 BIG/SMALL/SMALL), polars는 입력 순서 고정.
`_statistics/_target_decoy_q.py:60` 의 `sort_values("PEP")` 가 unstable quicksort라
**PEP 동점에서 q-value가 행 순서에 따라 달라집니다** → polars 전환이 이걸 개선합니다.

---

## E. 테스트 — 지우면 oracle도 같이 죽음

**polars↔pandas parity 테스트는 딱 1개**:
`tests/test_read_write_maxquant_null_scan.py::test_maxquant_tmt_polars_matches_pandas_with_null_scan`.
**Sage / DIA-NN / FragPipe / DELPI 는 parity 테스트가 0개** — pandas 분기가 사실상 스펙입니다.
그런데 §A에서 확인했듯 그 4개 리더에 심각 결함이 몰려 있습니다. 우연이 아닙니다.

**sparse↔dense parity 5개**(`tests/test_preprocessing_tmt.py:56/77/120/132/142`)는
**전부 `split_tmt(..., sparse=False)` 를 직접 호출** → `_build_block_diagonal_dense` 를 지우면
oracle이 같이 죽습니다.
반면 `tests/test_tools_sparse_layer.py`(4개)는 dense 레퍼런스를 손으로 만들어서 **살아남습니다** —
이게 따라야 할 패턴입니다.

**`read_diann(sparse=True)` 는 테스트가 0개** — `SparseQuantFrame` 도 `read_diann(sparse=)` 도
`tests/` 어디에도 없습니다. `_diann.py:126-136` COO 빌더, `SparseQuantFrame.anndata_x`,
`_base_reader.py:521-528` sparse 분기 전부 미검증. **이게 유일 경로가 될 예정입니다.**

---

## F. 검증해서 깨끗한 것 (되짚을 필요 없음)

- **`aggregate_features_by_group`**: 랜덤 900회(median/mean/sum × 결측률 0/30/70/95% ×
  명시적 0 × all-NaN 그룹), pandas dense groupby 기준 **0 불일치**.
- **영속성/슬라이싱 전부 안전**: `to_observed_sparse` → 행/열 슬라이스 → `copy()` →
  `tocsr/tocsc/T/astype` → `sp.vstack` → AnnData view → MuData 슬라이스 →
  **h5ad 왕복 → h5mu 왕복** 전부 nnz·값 보존, 명시적 0 포함.
  `MuData()` 생성 / `.update()` / `push_obs` 도 densify 안 함.
  *유일한 예외*: **sparse+sparse 덧셈이 명시적 0을 제거** (현재 미사용, 향후 landmine).
- **DIA-NN polars vs pandas**: `.X` bit-identical, `var` 값·dtype 완전 일치.
- **`stripped_peptide`**: `re.findall(r"([A-Z]+)|(\[\+\d+\.\d+\])")` ≡ `replace_all("[^A-Z]","")`.
  16개 적대적 케이스 0 불일치. **두 파일 모두 lookaround 없음** — Rust regex 우려 해당 없음.
- **`parse_uniprot_accession_group`** via `_map_unique` ≡ `replace_strict`, 12케이스 0 불일치.
- **MaxQuant/DELPI identification 변환**, **MaxTmtReader `_extract_quant_from_raw`**(수정 후),
  **FragPipe identification**, **`LfqSageReader`/`LfqFragPipeReader` quant**: 전부 동일.
- **실제 Sage 픽스처 end-to-end**(TMT+LFQ): shape/var_names/obs_names/dtype/`X`/`varm` 동일,
  차이는 §D-4 (1-ULP) 뿐.
- **dense/sparse 수치 동등성 end-to-end**: `normalise`, `scale_data`, `log2_transform`,
  `collapse_obs`, `to_peptide`, `to_protein(median)`, `correct_batch_effect`, `to_readable`,
  `PlotData._get_data`, `tl.corr`, `tl.pca`, `tl.run_de(limma)`.

---

## G. dtype / 메모리 — 결정이 필요한 항목

### G-1. sparse가 `.X` 를 float64 → float32 로 말없이 다운캐스트 · CONFIRMED

```
polars/dense  vs pandas/dense : bit-identical ✔   (polars 포팅 자체는 깨끗)
polars/sparse vs pandas/dense : float32 vs float64, maxdiff 1.9e-05, NaN 패턴 동일 ✔
```
원인: `.X` 생성 4곳 중 **dense 동일레벨 분기만** `.astype(np.float32)` 누락
(`_base_reader.py:533`; `:553/:570/:582` 및 `_diann.py:134` 는 float32).
**sparse 쪽이 오히려 msmu 관례와 일치** — 포팅 버그가 아니라 기존 불일치의 노출.
권장: `_base_reader.py:533` 도 float32로 지금 통일 → 전환일에 수치가 안 바뀜.

`_blockdiag.py:173` `to_dense_df` 의 `.astype(matrix.dtype)` 도 (a) sparse에서 float32 반환,
(b) 비-float `.X` 면 **NaN 채움이 0으로 바뀜** — 침묵 0을 막는 헬퍼 안의 침묵 0 경로.

### G-2. sparse의 메모리 이득이 기본 워크플로에서 상쇄됨 · CONFIRMED

```
read              [psm] X=csr_matrix   nan-pattern=True  maxdiff=6.7e-03
log2_transform    [psm] X=csr_matrix   nan-pattern=True  maxdiff=9.3e-07
normalise(median) [psm] X=ndarray      nan-pattern=True  maxdiff=1.8e-06   <- 여기서 dense 복귀
to_peptide    [peptide] X=ndarray      nan-pattern=True  maxdiff=1.8e-06
```
sparse 표현이 **read + log2 두 단계만** 유지됩니다. 게다가:
- `_summarisation.py:642-646` — 마스크가 블록대각 전체를 densify. **TMT `to_peptide` 기본값
  `purity_threshold=0.7` 이 항상 이 분기**. 결과가 float64라 legacy dense(float32) 대비 **피크 2배**.
- `_summarisation.py:621` `_make_rank_mask` 가 또 densify. **`to_protein` 기본 `top_n=3`** →
  **동시에 dense 사본 2개**, 최대 4배.
- `normalise`/`scale_data`/`correct_batch_effect`/`collapse_obs`/`adjust_ptm_by_protein`/
  `to_*` 3종/`read_flashlfq`/`_base_reader` 3분기 — 전부 dense를 `.X` 에 되씀.

`_normalise.py:158-161` 에 "follow-up" 으로 의도가 명시돼 있으니 버그는 아닙니다. 다만
**메모리가 목적이라면 리더만 sparse화해서는 목표 미달성**입니다.

---

## H. 정리 대상 (죽은 코드)

**이미 dead** (grep 확인, `.venv`/`build` 제외):
`DiannReader._split_merged_identification_quantification`(`_diann.py:85`, 테스트만 호출) ·
`DelpiReader._split_merged_identification_quantification`(`_delpi.py:64`, 호출 0) ·
`SageReader._label_decoy`(`_sage.py:64`) · `_label_possible_contaminant`(`:71`) ·
`_read_config_file`(`:81`, 실행되면 `AttributeError`) ·
`SearchResultReader._validate_search_outputs`(`_base_reader.py:341`, 유일 호출부가 `:604` 주석 처리) ·
`polars_native_enabled`(`_base_reader.py:43`) · **`to_observed_sparse`(`_blockdiag.py:97`,
`__all__` 에 있는데 호출 0)** · `MaxDiaReader` · `DiannProteinGroupReader` ·
`_base_reader.py:231-236` 의 도달 불가 분기.

**계산 후 폐기**: `DiannReader.missed_cleavages` — `used_feature_cols` 에 없어서
`_base_reader.py:442` 에서 버려집니다. **두 분기 모두**(`_diann.py:154` regex,
`:193-195` count_matches 2회) 핫패스에서 헛일 중.

**pandas 삭제 후 dead가 됨**: `_map_unique`(`_base_reader.py:318`) ·
`SageReader._extract_scan_number`(`_sage.py:78`) · `FragPipeReader._label_decoy`(`_fragpipe.py:105`).
`_strip_filename` 은 **생존** (`_sage.py:266` 에서 양쪽 경로 사용).

**문서/API**: `set_polars_reader` 는 **공개 API**(`msmu.io`)라 제거 시 breaking change.
`docs/`·`README`·`CHANGELOG` 에는 polars/sparse 언급이 **아예 없음** — 마이그레이션이 통째로 미문서화.
`fastparquet` 은 `_meta.py:203` 이 아직 쓰므로 제거 불가.

---

## I. 삭제 전 체크리스트

**즉시 (기본 경로 침묵 결함)**
1. `_sage.py:133` — 스캔 파싱 실패를 null이 아니라 명시적 에러로 (§A-1)
2. `_sage.py:202` — `.cast(pl.Int64).cast(pl.Utf8)` 추가, FragPipe와 동일하게 (§A-2). 1줄
3. `_delpi.py:86` — MaxQuant와 동일 수정 + DELPI 테스트 신설 (§A-3)
4. `_sage.py:142` — `.fill_null(False)` (§A-4)
5. `_fragpipe.py:171` / `_maxquant.py:23` — null 유지 (§A-5)
6. `_tmt.py:91` + `:115` — `toarray()`/`to_df()` → `dense_block()` (§C-1)

**삭제 전 필수**
7. 파서 기본값 고정 (§B-5) + B-3 처리 방식 결정
8. `pl.concat` → `diagonal_relaxed` (§B-4)
9. DataFrame 리스트 입력 — 지금 깨져 있으므로 고치거나 명시적으로 거부 (§B-6)
10. null 그룹 키 정책 통일 — `"null"`/`nan` 유령 그룹 대신 raise (§A-6, §C-2)
11. `PtmSummarisationPrep.prep` sparse 대응 + `median_polish`/`directlfq` sparse 경로 (§C-3)
12. `_has_quant_values` 의 `nnz==0` (§C-4) · `SparseQuantFrame` 가드 (§C-5)
13. `.X` dtype 통일 + `to_dense_df` 의 `astype` 제거 (§G-1)

**전략 — 순서가 중요**
14. **골든 픽스처를 먼저 박고 나서 지울 것** (§E). parity 테스트가 유일한 oracle이고,
    특히 Sage/DIA-NN/FragPipe/DELPI는 parity 테스트가 0개인데 심각 결함이 거기 몰려 있습니다.
    `tests/test_tools_sparse_layer.py` 방식(레퍼런스를 손으로 구성)이 따라야 할 패턴입니다.
15. **`read_diann(sparse=True)` 테스트 신설** — 유일 경로가 될 코드가 지금 완전 미검증 (§E).
16. 메모리가 목적이면 §G-2 — `_summarisation` 마스킹을 sparse-native로
    (마스킹은 CSC 패턴에서 컬럼 빼기일 뿐).

---

## 부록 — 별건(포팅 무관) 기존 버그

- `_tools/_correlation.py:41-43` — `obsp["X_corr"]` 를 로컬 `.copy()` 에 써서 `corr()` 결과가 버려짐
- `_fragpipe.py:123, :130` — 이 venv의 pandas 3.0에서 FragPipe pandas 분기가 실행 불가
- pandas 3.0 Arrow-string 기본값 때문에 `write_h5ad`/h5mu 쓰기가 깨짐
  (`pd.set_option("future.infer_string", False)` 로 우회)
- `_base_reader.py:104-105` — `MuDataInput` docstring이 없는 필드(`search_result`)를 설명
