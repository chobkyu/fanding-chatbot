from langchain.tools import Tool
from functions.search_fanding_site import search_fanding_site

search_tool = Tool.from_function(
    func = search_fanding_site,
    name = "search_fanding_site",
    description = "실시간 정보나 인기 크리에이터 등 팬딩 사이트에서 직접 확인해야 하는 질문에 사용하세요. 팬딩 웹사이트에서 실시간 정보를 검색할 때 사용합니다. 예: 인기 있는 크리에이터, 최신 소식 등."
)
