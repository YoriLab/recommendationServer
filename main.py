from pydoc import resolve
from fastapi import FastAPI
from typing import Union, Optional
from konlpy.tag import Okt
import pickle
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

## HOST 명시가 spring이 받게 하는 해결책?
app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
]

# CORS 문제일 가능성 다
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



okt = Okt()
with open('dish_vectorizer', 'rb') as f:  
  dish_vectorizer = pickle.load(f)
with open('soup_vectorizer', 'rb') as f:  
  soup_vectorizer = pickle.load(f)
with open('dish_bkmean', 'rb') as f:  
  dish_bkmean = pickle.load(f)
with open('soup_bkmean', 'rb') as f:  
  soup_bkmean = pickle.load(f)
with open('dish_cat_names.json', 'r', encoding='utf-8') as f:
  dish_cat_names = json.load(f)
with open('soup_cat_names.json', 'r', encoding='utf-8') as f:
  soup_cat_names = json.load(f)



def to_vect(vectorizer, ing_str):
    return vectorizer.transform([' '.join(okt.nouns(ing_str))])



class RecommRequestBody(BaseModel):
  day: int
  dishCluster: int
  dishPointer: int
  soupCluster: int
  soupPointer: int


class RecommForMeal(BaseModel):
  whenToCook: int 
  dishNames: list[str] = []
  soupNames: list[str] = []

class RecommForDay(BaseModel):
  day: int 
  recomms: list[RecommForMeal] = []

class RecommResponseBody(BaseModel):
  dishPointer: int = 0
  soupPointer: int = 0
  recomms: list[RecommForDay] = [] # 아,점,저,아,점,저 ... (반찬 이름들, 국이름들)
  

class ClusterRequestBody(BaseModel):
  likes: list[str]


class ClusterResponseBody(BaseModel):
  dishCluster: int 
  soupCluser: int 




@app.get('/')
def root():
  return 'hel'

# 유저의 좋아하는 재료 리스트 넣으면 반찬 클러스터, 국 클러스터 반환
# 클러스터, 포인터, 몇일치 입력 받고 추천받기

@app.post('/users/cluster')
def get_clusters(requestBody: ClusterRequestBody):
  like_str = ' '.join(okt.nouns(' '.join(requestBody.likes)))
  dish_cluster_label = int(dish_bkmean.predict(to_vect(dish_vectorizer, like_str))[0])
  soup_cluster_label = int(soup_bkmean.predict(to_vect(soup_vectorizer, like_str))[0])
  response = ClusterResponseBody(dishCluster=dish_cluster_label, soupCluser=soup_cluster_label)
  return response


# 어떤 유저, 몇일 치 추천
@app.post('/users/recomm', response_model=RecommResponseBody)
def recommend(requestBody: RecommRequestBody):
  response = RecommResponseBody()

  dishCluster = str(requestBody.dishCluster)
  soupCluster = str(requestBody.soupCluster)
  day = requestBody.day
  dishPointer = requestBody.dishPointer
  soupPointer = requestBody.soupPointer


  for d in range(day):
    recommForDay = RecommForDay(day=d)
    for time_type in range(3):
      recommForMeal = RecommForMeal(whenToCook=time_type)
      for _ in range(3): # 반찬은 한 끼에 3개
        dish = dish_cat_names[dishCluster][dishPointer % len(dish_cat_names[dishCluster])]
        recommForMeal.dishNames.append(dish)
        dishPointer += 1
      for _ in range(1): # 국은 한 끼에 1개
        soup = soup_cat_names[soupCluster][soupPointer % len(soup_cat_names[soupCluster])]
        recommForMeal.soupNames.append(soup)
        soupPointer += 1

      recommForDay.recomms.append(recommForMeal)
    response.recomms.append(recommForDay)
  
  response.dishPointer = dishPointer
  response.soupPointer = soupPointer

  return response
