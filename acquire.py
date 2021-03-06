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

######### Matthew's function to get all the blogs!!! ###############
def codeup_blogs(urls):
    '''
    
    Description:
    -----------
    This is a helper function to collect the codeup blogs
    
    Parameters:
    ----------
    urls: list
        List of urls as strings
    
    '''
    # create blank dataframe to hold results
    blog_df = pd.DataFrame(columns=['title', 'body', 'category'])
    
    # loop that creates a list of urls
    for url in urls:
        
        # standard lines to create a soup variable 
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
        response = get(url, headers=headers)
        html = response.text
        soup = BeautifulSoup(html)
        
        # create a container
        container = soup.select('.container')[0]
        # within the container the h1 element = the title
        title = container.find('h1').text
        # collects the body text
        body = container.find(itemprop='text').text
        # collects the blog category
        category = container.find('rel').text
        # collects date
        #date = container.find(itemprop='datePublished').text
        # create a dictionary that holds the title and body
        container_dict = {
                    'title': title,
                    'body': body,
                    'category': category,
                    #'date': date
                        }
        # converts dictionary into a dataframe
        container_df = pd.DataFrame(container_dict,index=[0])
        # concats the container_df to the blog_df created earlier
        blog_df = pd.concat([blog_df, container_df], axis=0)
    
    # returns the blog_df with no duplicates and a reset index
    return blog_df.drop_duplicates().reset_index(drop=True)

def all_codeup_blogs():
    '''
    Description:
    -----------
    This function collects the title and body of ALL codeup blogs
    '''
    # hard coding the codeup blog website
    url = 'https://codeup.com/blog/'
    # standard lines to create a soup variable 
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    response = get(url, headers=headers)
    html = response.text
    soup = BeautifulSoup(html)
    # empty urls list
    urls = []
    # loop that creates the urls list
    for a in soup.select('a.jet-listing-dynamic-link__link', href=True):
        # add url to urls list
        urls.append(a['href'])
        # deletes duplicates and keeps unique values
        urls = list(set(urls))
    # runs the codeup_blogs with the urls created in the loop
    blogs_df = codeup_blogs(urls)
    # returns the blogs_df
    return blogs_df



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
    
categories = ["business", "sports", "technology", "entertainment", "science", "world"]

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
	return get_all_news_articles(categories)