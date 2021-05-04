from requests import get
from bs4 import BeautifulSoup
import pandas as pd

# Make a function that works on a single url
# Make sure your function has everything it needs inside (try to avoid globals)

def get_codeup_blog(url):
    
    # Set the headers to show as Netscape Navigator on Windows 98, b/c I feel like creating an anomaly in the logs
    headers = {"User-Agent": "Mozilla/4.5 (compatible; HTTrack 3.0x; Windows 98)"}

    # Get the http response object from the server
    response = get(url, headers=headers)
    
    soup = BeautifulSoup(response.text)
    
    title = soup.find("h1").text
    published_date = soup.time.text
    
    if len(soup.select(".jupiterx-post-image")) > 0:
        blog_image = soup.select(".jupiterx-post-image")[0].picture.img["data-src"]
    else:
        blog_image = None
        
    content = soup.select(".jupiterx-post-content")[0].text
    
    output = {}
    output["title"] = title
    output["published_date"] = published_date
    output["blog_image"] = blog_image
    output["content"] = content
    
    return output


def get_blog_articles(urls):
    # List of dictionaries
    posts = [get_codeup_blog(url) for url in urls]
    
    return pd.DataFrame(posts)


def acquire_codeup_blog():
	urls = [
	    "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
	    "https://codeup.com/data-science-myths/",
	    "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
	    "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
	    "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/"
	]

	return get_blog_articles(urls)



# Inshort article
def get_article(article, category):
    # Attribute selector
    title = article.select("[itemprop='headline']")[0].text
    
    # article body
    content = article.select("[itemprop='articleBody']")[0].text
    
    output = {}
    output["title"] = title
    output["content"] = content
    output["category"] = category
    
    return output




def get_articles(category):
    """
    This function takes in a category as a string. Category must be an available category in inshorts
    Returns a list of dictionaries where each dictionary represents a single inshort article
    """
    base = "https://inshorts.com/en/read/"
    
    # We concatenate our base_url with the category
    url = base + category
    
    # Set the headers to show as Netscape Navigator on Windows 98, b/c I feel like creating an anomaly in the logs
    headers = {"User-Agent": "Mozilla/4.5 (compatible; HTTrack 3.0x; Windows 98)"}

    # Get the http response object from the server
    response = get(url, headers=headers)

    # Make soup out of the raw html
    soup = BeautifulSoup(response.text)
    
    # Ignore everything, focusing only on the news cards
    articles = soup.select(".news-card")
    
    output = []
    
    # Iterate through every article tag/soup 
    for article in articles:
        
        # Returns a dictionary of the article's title, body, and category
        article_data = get_article(article, category) 
        
        # Append the dictionary to the list
        output.append(article_data)
    
    # Return the list of dictionaries
    return output
    

def get_all_news_articles(categories):
    """
    Takes in a list of categories where the category is part of the URL pattern on inshorts
    Returns a dataframe of every article from every category listed
    Each row in the dataframe is a single article
    """
    all_inshorts = []

    for category in categories:
        all_category_articles = get_articles(category)
        all_inshorts = all_inshorts + all_category_articles

    df = pd.DataFrame(all_inshorts)
    return df


def acquire_news_articles():
	categories = ["business", "sports", "technology", "entertainment", "science", "world"]
	return get_all_news_articles(categories): 