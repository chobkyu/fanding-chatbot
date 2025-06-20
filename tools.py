from langchain.tools import Tool
from functions.search_fanding_site import search_fanding_site

search_tool = Tool.from_function(
    func = search_fanding_site,
    name = "search_fanding_site",
    description = "팬딩 웹사이트에서 실시간 정보를 검색할 때 사용합니다. 예: 인기 있는 크리에이터, 최신 소식 등."
)
