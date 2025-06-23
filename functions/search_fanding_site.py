import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  # ✅ Service import 추가
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import random

def search_fanding_site(query: str) -> str:
    """
    팬딩 웹사이트에서 실시간 데이터를 크롤링합니다.
    """
    print("팬딩 웹사이트 실시간 데이터 수집 크롤링")
    driver = None
    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://fanding.kr/")

        # 명시적으로 특정 요소가 로드될 때까지 최대 10초 기다림
      
        WebDriverWait(driver, 20).until(
            lambda d: any(
                e.text.strip() for e in d.find_elements(By.CLASS_NAME, "name__text__marquee")
            )
        )
                   

        # 페이지 소스 확인용 (문제 디버깅 시)
        # with open("page.html", "w", encoding="utf-8") as f:
        #     f.write(driver.page_source)
        
        driver.implicitly_wait(5)

        html = driver.page_source
        # print(html)
        soup = BeautifulSoup(html, "html.parser")

        creators = soup.select(".name__text__marquee")
        # print(creators)

        names = [c.text.strip() for c in creators if c.text.strip()]

        if not names:
            return "❌ 크리에이터 이름을 찾을 수 없습니다."

        unique_texts = list(set(names))
        data = random.sample(unique_texts, min(15,len(unique_texts)))

        print(data)
        return f"현재 인기 있는 크리에이터는: {', '.join(data)}입니다."
    except Exception as e:
        print(e)
        return f"실시간 검색에 실패했습니다: {str(e)}"
    except TimeoutException:
        print("❌ 요소 로딩 실패 - 타임아웃 발생")
        driver.quit()
        return "페이지 로딩 실패"
    finally:
        if driver:
            driver.quit()


# ✅ 여기서 직접 실행
if __name__ == "__main__":
    search_fanding_site()