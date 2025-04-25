"""
Module for scraping El País hemeroteca pages by date using Selenium.
Includes setup of WebDriver, page loading with retries, article extraction,
and date range iteration for multiple years.
"""
import datetime
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


def setup_driver(headless=False):
    """
    Configure and return a Selenium Chrome WebDriver instance.

    Parameters
    ----------
    headless : bool, optional
        Whether to run Chrome in headless mode (default: False).

    Returns
    -------
    webdriver.Chrome
        Configured WebDriver instance.
    """
    options = Options()
    options.add_argument("--user-data-dir=/Users/pablochamorro/Library/Application Support/Google/Chrome/Profile 6")
    options.add_argument("--profile-directory=Default")
    # if headless:
    #     options.add_argument("--headless=new")  # Run in headless mode
    options.add_argument("--start-maximized")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def safe_get(driver, url, retries=3):
    """
    Attempt to navigate to a URL with retry logic on failure.

    Parameters
    ----------
    driver : selenium.webdriver
        The WebDriver instance to use for page navigation.
    url : str
        The URL to load.
    retries : int, optional
        Number of attempts before raising an exception (default: 3).

    Raises
    ------
    Exception
        Propagates the last exception if all retries fail.
    """
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} to load {url} failed: {e}")
            if attempt == retries - 1:
                raise


def scrape_articles(driver, web, date):
    """
    Scrape article titles and body text from a given El País archive page.

    Parameters
    ----------
    driver : selenium.webdriver
        The WebDriver instance with an active session.
    web : str
        URL of the El País hemeroteca page for a specific date.
    date : datetime.date
        The date corresponding to the archive page.

    Returns
    -------
    list of dict
        A list of dictionaries, each with keys 'Title' and 'Body_Text'.
    """
    news_list = []
    try:
        safe_get(driver, web)
        time.sleep(1)  # Allow page to load

        # Accept cookies, if present
        try:
            cookies_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div/div/div/div/div[2]/button[2]'))
            )
            cookies_button.click()
            time.sleep(0.5)
        except Exception:
            print("No cookies button found or already accepted.")

        # Locate article links
        articles = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//article//h2/a"))
        )   

        for article in articles:
            try:
                title = article.text
                link = article.get_attribute("href")

                # Open the article in a new tab
                driver.execute_script(f"window.open('{link}', '_blank');")
                time.sleep(1)

                # Switch to new tab
                driver.switch_to.window(driver.window_handles[-1])
                print("scraping: ", title)
                try:
                    # Accept cookies on article page if needed
                    try:
                        cookies_button2 = WebDriverWait(driver, 3).until(
                            EC.element_to_be_clickable((By.XPATH, '/html/body/div[6]/div/div/div/div[2]/div[1]/a'))
                        )
                        cookies_button2.click()
                        time.sleep(0.5)
                    except Exception:
                        print("No cookies button found or already accepted.")

                    # Locate article content container
                    content_div = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.XPATH,
                            '//div[@class="a_c clearfix" and @data-dtm-region="articulo_cuerpo"]'))
                    )
                    # Extract paragraphs
                    paragraphs = content_div.find_elements(By.XPATH, './/p[normalize-space()]')
                    body_text = ' '.join([p.get_attribute("textContent").strip() for p in paragraphs])

                    dict_news = {"Title": title, "Body_Text": body_text}
                    news_list.append(dict_news)
                    print(f"Scraped: {body_text}")

                except Exception as e:
                    print(f"Failed to scrape body text: {e}")
                finally:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

            except Exception as e:
                print(f"Error processing article: {e}")
                continue
    except Exception as e:
        print(f"Failed to scrape {web}: {e}")

    return news_list


def calculate_days_since(start_date):
    """
    Compute the number of days elapsed from a start date to today.

    Parameters
    ----------
    start_date : datetime.date
        The starting date.

    Returns
    -------
    int
        Number of days since start_date.
    """
    today = datetime.date.today()
    delta = today - start_date
    return delta.days


def calculate_days_in_year(year):
    """
    Determine the total number of days in a given calendar year.

    Parameters
    ----------
    year : int
        The calendar year (e.g., 2024).

    Returns
    -------
    int
        Number of days in the specified year.
    """
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)
    return (end_date - start_date).days + 1


def main():
    """
    Main execution: iterates over years and dates to scrape El País articles,
    storing results in a pandas DataFrame and saving to CSV.

    Steps:
      1. Set up WebDriver
      2. Loop through each year from 2024 to current year
      3. For each date, scrape articles and collect into list
      4. Save accumulated data to 'El_Pais_news.csv'
    """
    # Define the start and end year
    start_year = 2024
    end_year = datetime.date.today().year

    # Initialize the news list
    news = []

    # Setup WebDriver
    driver = setup_driver(headless=True)

    try:
        for year in range(start_year, end_year + 1):
            start_date = datetime.date(year, 1, 1)
            days_in_year = calculate_days_in_year(year)

            # Adjust days for current year
            if year == datetime.date.today().year:
                days_in_year = calculate_days_since(start_date) + 1

            print(f"Scraping {year}: {days_in_year} days to scrape.")
            for i in range(days_in_year):
                current_date = start_date + datetime.timedelta(days=i)
                web = f'https://static.elpais.com/hemeroteca/elpais/{current_date.strftime("%Y/%m/%d")}/m/portada.html'
                try:
                    news_list = scrape_articles(driver, web, current_date)
                    dict_news = {"date": current_date, "news": news_list}
                    news.append(dict_news)

                    # Print progress
                    progress = (i + 1) / days_in_year * 100
                    print(f"Year {year}: {progress:.2f}% completed.", end="\r")
                except Exception as e:
                    print(f"Error on {current_date}: {e}")

            print(f"\nYear {year} scraping completed.")
    finally:
        driver.quit()

    # Save data to CSV
    Pais_news = pd.DataFrame(news)
    Pais_news.to_csv("El_Pais_news.csv", index=False)
    print("The file has been saved successfully!")


if __name__ == "__main__":
    main()