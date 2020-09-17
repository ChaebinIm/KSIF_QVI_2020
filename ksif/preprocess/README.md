# 데이터 전처리 방법
## 소스 데이터 다운로드
(2019년 3월 1일 현재 기준)
### 파일명
190228.xlsx
### company sheet
- Code
    - KSE+KOSDAQ
    - All companies
    - Fiscal Month
    - Ex. Pref.S
    - List => ALL
    - Company
    - 종목명으로 오름차순 정렬
- Item
    - 최신 아이템 항목 즐겨찾기(company)
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 일간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: NONE
### etf sheet
- Code
    - KSE+KOSDAQ
    - All companies
    - Fiscal Month
    - Ex. Pref.S
    - List => ALL
    - ETF
- Item
    - etf
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 월간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: ALL
### benchmark sheet
- Code
    - benchmark
- Item
    - 최신 아이템 항목 즐겨찾기
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 월간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: ALL
### macro_daily sheet
- Code
    - 없음
- Item
    - macro_daily
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 일간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: NONE
### macro_monthly sheet
- Code
    - 없음
- Item
    - macro_monthly
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 월간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: ALL
### factor sheet
- Code
    - factor
- Item
    - factor
- Calendar
    - 자료기간: 1999-12-01 ~ 2019-02-28 (선택)
    - 자료주기: 월간
    - 통화: KRW
    - 금액: Default
    - 비영업일: Previous
    - 주말포함: ALL
## 데이터무결성 검증 1
0. 2019년 2월의 데이터 중 종가가 없는 회사코드와 회사명을 복사해
data_catalog 구글스프레드시트의 check 시트에 붙여넣고 unfiltered column이 TRUE인
회사들을 filter한다.
0. 위 회사들을 검색하여 상장폐지/합병 여부를 확인한다.
0. 상장폐지/합병의 여부에 따라 delisted/merged 시트를 업데이트 한다.
## 전처리 코드
ksif/preprocess/core/__init__.py 파일을 실행하되, parameter로 엑셀의 파일명인 190228 입력
## 데이터업로드
0. 위 코드의 결과로 저장된 190228_company.csv, 190228_benchmark.csv, 190228_factor.csv를 구글 드라이브에 업로드한다.
0. 파일 우클릭 - 공유 가능한 링크 가져오기 - 공유 설정 - 엑세스 권한이 있는 사용자 - 링크 공유 사용 중 - 외부 엑세스 허용 체크
0. 두 파일의 URL 중 key(예를 들어 https://drive.google.com/file/d/17g0rNXV-oplum164qDBCII8bGCkeaw9_/view?usp=sharing 중 17g0rNXV-oplum164qDBCII8bGCkeaw9_)를 data_catalog 구글스프레드시트의 csv_files 시트에 날짜와 함께 저장한다.
