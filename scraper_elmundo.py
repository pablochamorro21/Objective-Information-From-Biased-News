"""
Module for scraping El Mundo political news articles by date from the hemeroteca,
with retry logic, infinite scroll handling, and saving results to CSV and JSON.
Supports incremental runs based on last processed date.
"""
import datetime
import os
import csv
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

def safe_get(driver, url, retries=3):
    """
    Attempt to load a URL in the Selenium driver with specified retries on failure.

    Parameters
    ----------
    driver : selenium.webdriver
        The WebDriver instance used to load pages.
    url : str
        The URL to navigate to.
    retries : int, optional
        Number of retry attempts on failure (default is 3).

    Raises
    ------
    Exception
        Any exception from the final failed attempt to load the page.
    """
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise

def scrape(web, date):
    """
    Scrape political news articles from a given El Mundo hemeroteca URL for a specific date.

    Opens the archive page, filters by section "Política", opens each article in a new tab,
    scrolls to load full content, extracts paragraphs, and returns a list of article dicts.

    Parameters
    ----------
    web : str
        The hemeroteca URL for the target date.
    date : str
        The date string to assign to scraped articles (format 'YYYY/MM/DD').

    Returns
    -------
    list of dict
        A list where each dict has keys: 'Title', 'Body_Text', and 'Date'.
    """
    news_list = []
    options = Options()
    #options.add_argument("--headless=new")  # Comenta esta línea para ver navegador en ejecución
    options.add_argument("--start-maximized")
    options.add_argument("--user-data-dir= /Users/pablochamorro/Library/Application Support/Google/Chrome/Profile 6")
    options.add_argument("--profile-directory= Default")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(200)
    driver.implicitly_wait(10)

    # Navegar al enlace con reintentos
    try:
        safe_get(driver, web)
    except Exception as e:
        print(f"Error loading page {web}: {e}")
        driver.quit()
        return []  # Retorna lista vacía si no se pudo cargar la página

    # Esperamos que se cargue el contenedor principal
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '/html/body/main/div[5]/div/div[2]/div/div/div[1]'))
        )
    except Exception as e:
        print(f"Error loading the main container: {e}")
        driver.quit()
        return []

    # Localizamos artículos de política
    political_news = []
    for n in range(1, 100):  # Aumentado a 100; ajusta si fuera necesario
        try:
            subject = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((
                    By.XPATH,
                    f'/html/body/main/div[5]/div/div[2]/div/div/div[1]/div[{n}]/article/div/div[2]/header/span'
                ))
            )
            if subject.text == "Política":
                political_news.append(n)
        except Exception:
            continue

    # Recorremos los artículos de política
    for n in political_news:
        try:
            title_element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((
                    By.XPATH,
                    f'/html/body/main/div[5]/div/div[2]/div/div/div[1]/div[{n}]/article/div/div[2]/header/a'
                ))
            )
            title = title_element.text
            link = title_element.get_attribute('href')

            # Abrimos el link en una nueva pestaña
            driver.execute_script(f"window.open('{link}', '_blank');")

            # Esperamos a que exista una nueva pestaña y cambiamos a ella
            WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) > 1)
            driver.switch_to.window(driver.window_handles[-1])

            # -----------------------------------------------
            # SCROLL INFINITO dentro del artículo (NUEVO)
            # -----------------------------------------------
            try:
                last_height_article = driver.execute_script("return document.body.scrollHeight")
                while True:
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)  # Espera a que cargue nuevo contenido en el artículo
                    new_height_article = driver.execute_script("return document.body.scrollHeight")
                    if new_height_article == last_height_article:
                        break
                    last_height_article = new_height_article
            except Exception as e:
                print(f"Error during scrolling in article {n}: {e}")

            # Ahora sí, esperamos a que se carguen los párrafos
            paragraphs = WebDriverWait(driver, 8).until(
                EC.presence_of_all_elements_located((By.XPATH, '//p[@data-mrf-recirculation="Links Párrafos"]'))
            )

            full_text = " ".join([p.text for p in paragraphs])

            dict_news = {"Title": title, "Body_Text": full_text, "Date": date}
            news_list.append(dict_news)

            print(title)
            print(full_text)

            # Cerramos la pestaña del artículo y volvemos a la principal
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        except Exception as e:
            print(f"Error processing article {n}: {e}")
            # Si falló y la pestaña está abierta, la cerramos
            if len(driver.window_handles) > 1:
                driver.close()
            driver.switch_to.window(driver.window_handles[0])
            continue

    driver.quit()
    return news_list

def get_last_processed_date(csv_file):
    """
    Obtain the most recent processed date from a CSV file, if it exists.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing a 'Date' column.

    Returns
    -------
    str or None
        The latest date string found in the file, or None if file is missing or empty.
    """
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            dates = [row["Date"] for row in reader]
            if dates:
                return max(dates)  # Última fecha en el archivo CSV
    return None

def calculate_days_and_years(start_date):
    """
    Calculate the number of days between today and a start date.

    Parameters
    ----------
    start_date : datetime.date
        The initial date from which to count.

    Returns
    -------
    int
        Number of days elapsed from start_date to today.
    """
    today = datetime.date.today()
    delta = today - start_date
    return delta.days

# -----------------------------------------------------
#                EJECUCIÓN PRINCIPAL
# -----------------------------------------------------
if __name__ == "__main__":
    # Configuración inicial
    start_date = datetime.date(2023, 2, 1)
    folder_name = "Mundo_from_23"
    os.makedirs(folder_name, exist_ok=True)
    csv_file = "Mundo_news_from_2023.csv"

    # Obtener la última fecha procesada
    last_date = get_last_processed_date(csv_file)
    if last_date:
        last_date = datetime.datetime.strptime(last_date, "%Y/%m/%d").date()
        # Empezar al día siguiente a la última fecha procesada
        start_date = last_date + datetime.timedelta(days=1)

    # Crear/abrir el archivo CSV para añadir datos
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', encoding='utf-8', newline='') as csvfile:
        fieldnames = ["Title", "Body_Text", "Date"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not csv_exists:
            writer.writeheader()

        # Calculamos el número de días a procesar
        num_days = calculate_days_and_years(start_date)
        for i in tqdm(range(num_days), desc="Processing days"):
            current_date = start_date + datetime.timedelta(days=i)
            formatted_date = current_date.strftime('%Y/%m/%d')
            web = f'https://www.elmundo.es/elmundo/hemeroteca/{formatted_date}/noticias.html'

            try:
                news_list = scrape(web, formatted_date)
            except Exception as e:
                print(f"Error scraping {formatted_date}: {e}")
                continue

            # Guardamos artículos en CSV y en JSON
            for idx, article in enumerate(news_list, start=1):
                writer.writerow(article)
                file_name = f"{folder_name}/{formatted_date.replace('/', '-')}_{idx}.json"
                with open(file_name, 'w', encoding='utf-8') as json_file:
                    json.dump(article, json_file, ensure_ascii=False, indent=4)

    print("THE FILES AND CSV HAVE BEEN SAVED CORRECTLY!")
