from selenium import webdriver
from random import randint
from time import sleep
from functools import wraps


# Helper function to set User-Agent
def set_user_agent(user_agent_str):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument(f"user-agent={user_agent_str}")
    return chrome_options


# Helper function to map human behavior to random delay
def map_human_behavior_to_delay(action=None):
    action = 'clicking' if action is None else action
    behavior_to_delay = {
        'fast_typing': (0, 1),
        'slow_typing': (1, 3),
        'mouse_move': (2, 4),
        'shortcut': (0.1, 0.3),
        'page_scroll': (1, 2),
        'clicking': (1, 2),
    }
    return behavior_to_delay.get(action, (1, 2))


# Helper function to add random delay
def random_delay(min_seconds=0.1, max_seconds=1.0, *, mode=None):
    if mode:
        min_seconds, max_seconds = map_human_behavior_to_delay(mode)
    sleep(randint(min_seconds, max_seconds))


# Helper function to maintain cookies
def maintain_cookies(driver, cookies):
    for cookie in cookies:
        driver.add_cookie(cookie)


def human_browse(user_agent_str='default', action='clicking', cookies=True):
    input('yoyoyoy')
    if user_agent_str == 'default':
        user_agent_str = 'chrome117.0.0.0'

    if user_agent_str.startswith('chrome'):
        version = user_agent_str[len('chrome') :]
        user_agent_str = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like"
            f" Gecko) Chrome/{version} Safari/537.36"
        )

    def decorator(func):
        @wraps(func)
        def wrapper(*, url, **kwargs):
            try:
                chrome_options = set_user_agent(user_agent_str)
                chrome_options.headless = False  # Run in non-headless mode
                input(chrome_options)
                driver = webdriver.Chrome(options=chrome_options)

                # Navigate to the URL
                driver.get(url)

                # Add cookies if available
                if cookies:
                    maintain_cookies(driver, cookies)

                # Mimic human delay based on the action
                min_seconds, max_seconds = map_human_behavior_to_delay(action)
                random_delay(min_seconds, max_seconds)

                # Run the function with the driver as an argument
                func(driver=driver, url=url, **kwargs)

            except Exception as e:
                input(f"An error occurred: {e}")

            finally:
                # Close the driver
                driver.quit()

        return wrapper

    return decorator


# Example usage
if __name__ == '__main__':
    url = "http://example.com"
    user_agent_str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/89.0.4389.82 Safari/537.3"
    )
    action = (  # Replace this with the type of action you want to simulate
        'mouse_move'
    )
    example_cookies = [{"name": "test", "value": "test_value"}]

    human_browse(url, user_agent_str, action, cookies=example_cookies)


def get_zoom_html(url, driver, zoom_level):
    driver.get(url)

    # Give the page time to load
    time.sleep(3)

    # Create an ActionChains object
    actions = ActionChains(driver)

    # Move focus to the body of the page to execute zoom
    actions.move_to_element(driver.find_element_by_tag_name('body'))

    # Perform zoom out (Ctrl and -)
    for _ in range(7):  # Zoom out three times to achieve something close to 25%
        actions.key_down(Keys.CONTROL).send_keys(Keys.SUBTRACT).key_up(
            Keys.CONTROL
        )

    # Execute the actions
    actions.perform()

    # Allow time for the zoom action to take effect
    time.sleep(2)

    # Capture the HTML content
    html_content = driver.page_source

    # Close the browser
    driver.quit()

    # Now you can process the HTML content as needed
    print(html_content)
