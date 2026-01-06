#pip install selenium pandas webdriver-manager

import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 1. 필수 라이브러리 설치 (필요시 주석 해제 후 실행)
# !pip install selenium webdriver-manager

import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 데이터 리스트 ---
lsi_ids = [103537, 119420, 125100, 126914, 172505, 175133, 180744, 183954, 184833, 199536, 203319, 203336, 205871, 206701, 208466, 208499, 209898, 210616, 218455, 224121, 227139, 233367, 234267, 234675, 235847, 237465, 237467, 239967, 240833, 241563, 241703, 242023, 243313, 244041, 246999, 249317, 249869, 251721, 252715, 253665, 254697, 255029, 255883, 256073, 257855, 257869, 258101, 258985, 259093, 259255, 259439, 260757, 260821, 261143, 261447, 262725, 263063, 263109, 263169, 263325, 263583, 263819, 263895, 263903, 264095, 264797, 265597, 266187, 266255, 266527, 266609, 266687, 266791, 267289, 267353, 267385, 267389, 267415, 267429, 267431, 267745, 268255, 268883, 269175, 269835, 269841, 269967, 270039, 270215, 270349, 270367, 270369, 270393, 270399, 270401, 270403, 270405, 270407, 270409, 270413, 270789, 270793, 270795, 270797, 270805, 270807, 270811, 271009, 271111, 271113, 272453, 272519, 272577, 272611, 273463, 273847, 273849, 273895, 273897, 273901, 273903, 275221, 275225, 275227, 276021, 276031, 276649, 276651, 276653, 276655, 276657, 276659, 276663, 276679, 276683, 276865, 276867, 276869, 276871, 276873, 276877, 276879, 276889, 276891, 276893, 276895, 276897, 276899, 276901, 277429, 277435, 277441, 277445, 277451, 277457, 277461, 277475, 277903, 278819, 278821, 278825, 278829, 278831, 278833, 278837, 278839, 278843, 278845, 278847, 278849, 278861, 279063, 279207, 279233, 279255, 279423, 279627, 279631, 279669, 279671, 279689, 279697, 279699, 279707, 279741, 279759, 279833, 279851, 279933, 280225, 280255, 280269, 280291, 280293, 280315, 280333, 280393, 280447, 280495, 280521, 280637, 281153, 281155, 281165, 281167, 281169, 281179, 281233, 281235, 281585, 281587, 281589, 281591, 281595, 281629, 281631, 281633, 281635, 281637, 281843, 281919, 281921, 281925, 281927, 281929, 281941, 281947, 281949, 281951, 281953, 281955, 281959, 281961, 282119, 282223, 282237, 282417, 282419, 282493, 282573, 282763]
output_file = "사회복지_법령_전체.txt"

# --- 브라우저 설정 ---
chrome_options = Options()
chrome_options.add_argument("--headless") 
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# --- 실행 ---
try:
    # 'a' 모드 대신 'w' 모드로 열어 실행 시마다 파일을 새로 작성합니다.
    with open(output_file, "w", encoding="utf-8") as f:
        total = len(lsi_ids)
        for i, lsi_id in enumerate(lsi_ids):
            url = f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={lsi_id}"
            
            try:
                driver.get(url)
                
                # contentBody 요소 대기
                wait = WebDriverWait(driver, 10)
                body = wait.until(EC.presence_of_element_located((By.ID, "contentBody")))
                
                # 텍스트 추출 (앞뒤 공백 제거)
                law_content = body.text.strip()
                
                # 데이터 기록: [내용] + [두 줄 개행]
                f.write(law_content)
                f.write("\n\n\n") 
                
                print(f"[{i+1}/{total}] 크롤링 완료: {lsi_id}")
                
            except Exception as e:
                print(f"[{i+1}/{total}] 실패 ({lsi_id}): {e}")
            
            # 서버 부하 방지용 (필요에 따라 조절)
            time.sleep(1.0)

finally:
    driver.quit()
    print(f"\n✅ 모든 작업이 끝났습니다. 파일명: {output_file}")