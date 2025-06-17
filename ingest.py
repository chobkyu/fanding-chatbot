# ingest.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path
import os

# .env 로드
load_dotenv()

# 크롤링할 문서 URL들
URLS = [
    "https://docs.channel.io/user_guide/ko/articles/2025%EB%85%84-4%EC%9B%94-4%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-cbfbde35,"
    "https://docs.channel.io/user_guide/ko/articles/2025%EB%85%84-3%EC%9B%94-5%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-f1a85632",
    "https://docs.channel.io/user_guide/ko/articles/2025%EB%85%84-3%EC%9B%94-2%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-7a8d0a00",
    "https://docs.channel.io/user_guide/ko/articles/2025%EB%85%84-2%EC%9B%94-5%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-726e4428",
    "https://docs.channel.io/user_guide/ko/articles/2025%EB%85%84-2%EC%9B%94-4%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-659c85b7",
    "https://docs.channel.io/user_guide/ko/articles/2024%EB%85%84-10%EC%9B%94-5%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-6c84cf8f",
    "https://docs.channel.io/user_guide/ko/articles/2024%EB%85%84-9%EC%9B%94-4%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-e63812a1",
    "https://docs.channel.io/user_guide/ko/articles/2024%EB%85%84-9%EC%9B%94-1%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-71141931",
    "https://docs.channel.io/user_guide/ko/articles/2024%EB%85%84-8%EC%9B%94-5%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-1e17444d",
    "https://docs.channel.io/user_guide/ko/articles/2024%EB%85%84-8%EC%9B%94-3%EC%A3%BC%EC%B0%A8-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8-%EB%85%B8%ED%8A%B8-d5c37801",
    "https://docs.channel.io/user_guide/ko/articles/FAQ-%EC%9E%90%EC%A3%BC-%EB%AC%BB%EB%8A%94-%EC%A7%88%EB%AC%B8-26d00117",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8C%AC%EB%94%A9%EC%9D%B4%EB%9E%80-3804e6c8",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8C%AC%EB%94%A9-%EC%A0%91%EC%86%8D-%EB%B0%A9%EB%B2%95-15e7e442",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8C%AC%EB%94%A9-%EC%95%B1-%EC%95%8C%EB%A6%BC-%EC%84%A4%EC%A0%95-ec0aaa4b",
    "https://docs.channel.io/user_guide/ko/articles/%ED%81%AC%EB%A6%AC%EC%97%90%EC%9D%B4%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-5da30947",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9A%8C%EC%9B%90-%EA%B0%80%EC%9E%85-2930bda0",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A1%9C%EA%B7%B8%EC%9D%B8-c7acaf29",
    "https://docs.channel.io/user_guide/ko/articles/%ED%94%84%EB%A1%9C%ED%95%84-%EC%84%A4%EC%A0%95-d64a6623",
    "https://docs.channel.io/user_guide/ko/articles/%EC%84%B1%EC%9D%B8%EB%B3%B8%EC%9D%B8-%EC%9D%B8%EC%A6%9D-003dbfd8",
    "https://docs.channel.io/user_guide/ko/articles/%EC%9D%B4%EB%A9%94%EC%9D%BC-%EB%B3%80%EA%B2%BD-5de62490",
    "https://docs.channel.io/user_guide/ko/articles/%EB%B9%84%EB%B0%80%EB%B2%88%ED%98%B8-%EC%84%A4%EC%A0%95-%EB%B0%8F-%EB%B3%80%EA%B2%BD-e7a224b4",
    "https://docs.channel.io/user_guide/ko/articles/%EA%B8%B0%EA%B8%B0-%EB%93%B1%EB%A1%9D-61a42038",
    "https://docs.channel.io/user_guide/ko/articles/%EC%84%B1%EC%9D%B8-%EC%BD%98%ED%85%90%EC%B8%A0-%EC%B6%94%EC%B2%9C-%EC%88%A8%EA%B8%B0%EA%B8%B0-5e13dfec",
    "https://docs.channel.io/user_guide/ko/articles/%EA%B3%84%EC%A0%95-%ED%83%88%ED%87%B4-3e318522",
    "https://docs.channel.io/user_guide/ko/articles/%EC%9C%A0%EC%A0%80%EC%9D%98-%ED%83%80%EC%9E%85-db3e9c9c",
    "https://docs.channel.io/user_guide/ko/articles/%ED%81%AC%EB%A6%AC%EC%97%90%EC%9D%B4%ED%84%B0-%EC%B0%BE%EA%B8%B0-7709bf93",
    "https://docs.channel.io/user_guide/ko/articles/%EB%82%B4-%ED%81%AC%EB%A6%AC%EC%97%90%EC%9D%B4%ED%84%B0-3703b058",
    "https://docs.channel.io/user_guide/ko/articles/%EC%84%B1%EC%9D%B8-%ED%81%AC%EB%A6%AC%EC%97%90%EC%9D%B4%ED%84%B0-%ED%8E%98%EC%9D%B4%EC%A7%80-4fc0c223",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%A4%EB%B2%84%EC%8B%AD%EC%9D%B4%EB%9E%80-31f38cdb",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%A4%EB%B2%84%EC%8B%AD-%EA%B0%80%EC%9E%85%ED%95%98%EA%B8%B0-8cf534a3",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%A4%EB%B2%84%EC%8B%AD-%EB%B3%80%EA%B2%BD%ED%95%98%EA%B8%B0-be0fd7f8",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%A4%EB%B2%84%EC%8B%AD-%EC%A4%91%EB%8B%A8%ED%95%98%EA%B8%B0-ff094120",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%A4%EB%B2%84%EC%8B%AD-%EA%B0%B1%EC%8B%A0%ED%95%98%EA%B8%B0-03d4e78a",
    "https://docs.channel.io/user_guide/ko/articles/%EB%AC%B4%EB%A3%8C-%EA%B5%AC%EB%8F%85%EA%B6%8C-d5b70234",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A6%AC%EC%9B%8C%EB%93%9C%EB%9E%80-aaa3741a",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A6%AC%EC%9B%8C%EB%93%9C-%EC%9A%94%EC%B2%AD-%EC%84%A4%EB%AC%B8-ea0ad054",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A6%AC%EC%9B%8C%EB%93%9C-%EC%A7%80%EA%B8%89-%EB%82%B4%EC%97%AD-26790cba",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9E%88%ED%8A%B8%EB%9E%80-51272cd1",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9E%88%ED%8A%B8-%EC%B6%A9%EC%A0%84%ED%95%98%EA%B8%B0-7c5f3f1c",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9E%88%ED%8A%B8-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-b1002be0",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9E%88%ED%8A%B8-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-8e7d4cf2",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9E%88%ED%8A%B8-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-8e7d4cf2",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8F%AC%EC%8A%A4%ED%8A%B8-%EC%97%B4%EB%9E%8C-%EA%B6%8C%ED%95%9C-67753569",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8F%AC%EC%8A%A4%ED%8A%B8-%EA%B2%80%EC%83%89%ED%95%98%EA%B8%B0-499b0baf",
    "https://docs.channel.io/user_guide/ko/articles/%EC%98%81%EC%83%81-%ED%94%8C%EB%A0%88%EC%9D%B4%EC%96%B4-%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0-93e24603",
    "https://docs.channel.io/user_guide/ko/articles/%EC%98%A8%EB%9D%BC%EC%9D%B8-%EA%B0%95%EC%9D%98%EB%9E%80-5c8bd188",
    "https://docs.channel.io/user_guide/ko/articles/%EC%98%A8%EB%9D%BC%EC%9D%B8-%EA%B0%95%EC%9D%98-%EC%88%98%EA%B0%95%ED%95%98%EA%B8%B0-deb447b2",
    "https://docs.channel.io/user_guide/ko/articles/%EC%98%A8%EB%9D%BC%EC%9D%B8-%EA%B0%95%EC%9D%98-%EB%82%B4%EC%97%AD-362918b9",
    "https://docs.channel.io/user_guide/ko/articles/%EC%BF%A0%ED%8F%B0-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-5881c039",
    "https://docs.channel.io/user_guide/ko/articles/%EA%B0%95%EC%9D%98-%ED%94%8C%EB%A0%88%EC%9D%B4%EC%96%B4-%ED%99%9C%EC%9A%A9%ED%95%98%EA%B8%B0-d8027d0f",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%94%EC%8B%9C%EC%A7%80%EB%9E%80-6f937867",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%94%EC%8B%9C%EC%A7%80-%EB%B3%B4%EB%82%B4%EA%B8%B0-e4fc014d",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%94%EC%8B%9C%EC%A7%80-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-039d8074",
    "https://docs.channel.io/user_guide/ko/articles/%EB%A9%94%EC%8B%9C%EC%A7%80-%EC%82%AD%EC%A0%9C%ED%95%98%EA%B8%B0-33dff3fa",
    "https://docs.channel.io/user_guide/ko/articles/%EC%84%B1%EC%9D%B8-%EC%A0%84%EC%9A%A9-%EB%A9%94%EC%8B%9C%EC%A7%80-8977c80a",
    "https://docs.channel.io/user_guide/ko/articles/%EC%BB%A4%EB%AE%A4%EB%8B%88%ED%8B%B0%EB%9E%80-66d977b3",
    "https://docs.channel.io/user_guide/ko/articles/%EC%BB%A4%EB%AE%A4%EB%8B%88%ED%8B%B0-%EA%B8%80-%EC%9E%91%EC%84%B1-b2c3c31e",
    "https://docs.channel.io/user_guide/ko/articles/%EC%9D%B4%EB%B2%A4%ED%8A%B8%EB%9E%80-f2827deb",
    "https://docs.channel.io/user_guide/ko/articles/%EC%9D%B4%EB%B2%A4%ED%8A%B8-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-0709ad33",
    "https://docs.channel.io/user_guide/ko/articles/%EC%8A%A4%ED%86%A0%EC%96%B4%EB%9E%80-288b59bc",
    "https://docs.channel.io/user_guide/ko/articles/%EC%A3%BC%EB%AC%B8-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-ac01556a",
    "https://docs.channel.io/user_guide/ko/articles/%EB%B0%B0%EC%86%A1%EC%A7%80-%EB%B3%80%EA%B2%BD%ED%95%98%EA%B8%B0-d0ad868c",
    "https://docs.channel.io/user_guide/ko/articles/%EC%8A%A4%ED%86%A0%EC%96%B4-%EC%83%81%ED%92%88-%EA%B5%90%ED%99%98%EB%B0%98%ED%92%88-178da6db",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8E%80%EB%94%A9%EC%9D%B4%EB%9E%80-06508712",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8E%80%EB%94%A9-%EC%B0%B8%EC%97%AC%ED%95%98%EA%B8%B0-39173e54",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8E%80%EB%94%A9-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-7515cb89",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8E%80%EB%94%A9-%EC%9A%94%EC%B2%AD-%EC%84%A4%EB%AC%B8-e1420a96",
    "https://docs.channel.io/user_guide/ko/articles/%ED%8E%80%EB%94%A9-%EC%83%81%ED%92%88-%EA%B5%90%ED%99%98AS-aac48add",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9B%84%EA%B8%B0%EB%9E%80-8ba573e0",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9B%84%EA%B8%B0-%EC%9E%91%EC%84%B1%ED%95%98%EA%B8%B0-fbe877e3",
    "https://docs.channel.io/user_guide/ko/articles/%ED%9B%84%EA%B8%B0-%EC%88%98%EC%A0%95%EC%82%AD%EC%A0%9C%ED%95%98%EA%B8%B0-a2d2eb46",
    "https://docs.channel.io/user_guide/ko/articles/%EA%B2%B0%EC%A0%9C-%EC%88%98%EB%8B%A8-a78239ed",
    "https://docs.channel.io/user_guide/ko/articles/%EC%B9%B4%EB%93%9C-%EA%B0%84%ED%8E%B8-%EA%B2%B0%EC%A0%9C-5cc7c534",
    "https://docs.channel.io/user_guide/ko/articles/%EC%A0%95%EA%B8%B0-%EA%B2%B0%EC%A0%9C%EC%9E%90%EB%8F%99-%EA%B2%B0%EC%A0%9C-fcb2c145",
    "https://docs.channel.io/user_guide/ko/articles/%EA%B2%B0%EC%A0%9C-%EB%82%B4%EC%97%AD-%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0-ef6ed272",
    "https://docs.channel.io/user_guide/ko/articles/%ED%99%98%EB%B6%88-%EC%8B%A0%EC%B2%AD%ED%95%98%EA%B8%B0-76823b6a"
]

# 문서 로드
loader = WebBaseLoader(URLS)
docs = loader.load()

# 문서 나누기
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 임베딩 생성기
embedding = OpenAIEmbeddings()

# FAISS 벡터스토어 생성 및 저장
db = FAISS.from_documents(chunks, embedding)

# 저장 디렉토리 생성 및 저장
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
db.save_local(str(INDEX_DIR))

print("✅ FAISS 인덱스 생성 완료:", INDEX_DIR)