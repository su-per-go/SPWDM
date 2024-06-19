import re
from urllib.parse import urlparse
from Wappalyzer import Wappalyzer, WebPage
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing import Process


class PageFeatureExtra:
    def __init__(self, page_text_or_path, request_url, is_path=True):
        self.request_url = request_url
        if is_path:
            with open(page_text_or_path, "r", encoding="utf8") as f:
                self.page_text = f.read()
        else:
            self.page_text = page_text_or_path
        self.page_soup = BeautifulSoup(self.page_text, 'html.parser', from_encoding="utf8")
        self.feature_name = (
            "page_len", "page_tag_num", "page_tech_num", "page_domain_type_num", "page_like_req_domain_num",
            "page_unlike_req_domain_num", "page_hidden_tag_num", "page_have_copyright",
            "page_have_redirect"
        )
        self.link_ls = []
        for link in self.page_soup.find_all("a"):
            self.link_ls.append(link.get('href'))

    def page_len(self):
        return {"page_len": len(str(self.page_soup))}

    def page_tag_num(self):
        return {"page_tag_num": len(self.page_soup.findAll())}

    def page_tech_num(self):
        webpage = WebPage.new_from_html(self.page_text)
        wappalyzer = Wappalyzer.latest()
        return {"page_tech_num": len(wappalyzer.analyze(webpage))}

    def page_domain_type_num(self):
        return {"page_domain_type_num": len(self.link_ls)}

    def page_like_req_domain_num(self):
        return {"page_like_req_domain_num": self.get_like_domain_num()}

    def page_unlike_req_domain_num(self):
        return {"page_unlike_req_domain_num": len(self.link_ls) - self.get_like_domain_num()}

    def page_hidden_tag_num(self):
        # 查找具有style属性或CSS类中包含"hidden"的标签，以识别隐藏标签                       -----是否要考虑一下？
        hidden_tags = self.page_soup.find_all(
            lambda tag: tag.has_attr("style") and "display: none" in tag["style"] or tag.has_attr(
                "class") and "hidden" in tag["class"])
        return {"page_hidden_tag_num": len(hidden_tags)}

    def page_have_copyright(self):
        # copyright  ©
        elements = self.page_soup.find_all()
        for element in elements:
            if element.find(text=re.compile(r'copyright', re.I)) or element.find(text=re.compile(r'©', re.I)):
                return {"page_have_copyright": 1}
        return {"page_have_copyright": 0}

    def page_have_redirect(self):
        meta_tags = self.page_soup.find_all('meta', attrs={'http-equiv': 'refresh'})
        if meta_tags:
            return {"page_have_redirect": 1}
        meta_tags = self.page_soup.find_all('meta', attrs={'http-equiv': 'refresh'})
        if meta_tags:
            return {"page_have_redirect": 1}
        return {"page_have_redirect": 0}

    def get_like_domain_num(self):
        count = 0
        parsed_request_url = urlparse(self.request_url)
        for link in self.link_ls:
            if link is not None:
                if link.startswith("https://") or link.startswith("https://"):
                    parsed_link = urlparse(link)
                    domain_1 = set(parsed_link.netloc)
                    domain_2 = set(parsed_request_url.netloc)
                    try:
                        jaccard_similarity = len(domain_1.intersection(domain_2)) / len(domain_1.union(domain_2))
                    except ZeroDivisionError:
                        jaccard_similarity = 0
                    if jaccard_similarity > 0.68:
                        count += 1
                else:
                    count += 1
        return count

    def handle(self):
        feature_dict = {}
        for func_name in self.feature_name:
            function = getattr(self, func_name, None)
            feature_dict.update(function())
        return feature_dict

    @classmethod
    def get_default_feature(cls):
        return {
            'page_len': 0,
            'page_tag_num': 0,
            'page_tech_num': 0,
            'page_domain_type_num': 0,
            'page_like_req_domain_num': 0,
            'page_unlike_req_domain_num': 0,
            'page_hidden_tag_num': 0,
            'page_have_copyright': 0,
            'page_have_redirect': 0
        }


if __name__ == "__main__":
   pass