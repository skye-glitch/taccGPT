import motor.motor_asyncio
from models import (QaPair, 
                    Ranking,
                    RankingResults, 
                    Rankings, 
                    Answers, 
                    RankingResult)
from datetime import datetime

# client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://database:27017")
client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client["TACC_GPT"]
qa_collection = db["QA"]
ranking_collection = db['ranking']

async def add_one_qa_pair(qa_pair:QaPair):
    res = await qa_collection.insert_one(qa_pair.dict())
    return res

async def add_one_ranking(rankingResults:RankingResults):
    rankingResults_dict = {"prompt": rankingResults.prompt,
                           "date":datetime.now().strftime("%Y %m %d"),
                           'user':rankingResults.user}
    for rankingResult in rankingResults.rankings:
        key = f"rank {rankingResult.rank}"
        value = [val for val in rankingResult.rankingAnswers]
        rankingResults_dict[key] = value
        
    res = await ranking_collection.insert_one(rankingResults_dict)
    return res


async def fetch_all_qa_pairs():
    cursor = qa_collection.find({})
    qa_pairs = []
    async for qa_pair in cursor:
        qa_pairs.append(QaPair(**qa_pair))
    return qa_pairs

async def fetch_all_rankings():
    cursor = ranking_collection.find({})
    rankings = []
    async for ranking in cursor:
        prompt = ranking['prompt']
        date = ranking['date']
        user = ranking['user']
        num_answers = max([int(x.replace("rank ","")) if "rank " in x else 0 for x in ranking.keys()])
        answers = [ranking[f'rank {i+1}'] for i in range(num_answers)]
        rankings.append(Ranking(prompt=prompt, date=date, user=user, answers=answers))
    return rankings