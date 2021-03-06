import time

def wait_for(condition_function):
    """
    Sleep and retry every 0.1 seconds.
    """
    start_time = time.time()
    while time.time() < start_time + 5:
        if condition_function():
            return True
        else:
            time.sleep(0.1)

class WaitForPageLoad(object):
    """
    Wrapper to detect change of page before scraping content.
    """
    
    def __init__(self, browser):
        self.browser = browser

    def __enter__(self):
        self.old_page = self.browser.find_element_by_tag_name('html')

    def page_has_loaded(self):
        new_page = self.browser.find_element_by_tag_name('html')
        return new_page.id != self.old_page.id

    def __exit__(self, *_):
        wait_for(self.page_has_loaded)
