import motor.motor_asyncio
from models import (QaPair, 
                    Ranking,
                    RankingResults)
from datetime import datetime

import random
import json


client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://database:27017")
    

async def get_random_prompt():
    db = client["TACC_GPT"]
    first_prompts_collection = db['prompts']
    num_prompts = await first_prompts_collection.count_documents({})
    print(f"num_prompts: {num_prompts}")
    random_idx = random.randint(0, num_prompts)
    res = await first_prompts_collection.find_one({"_id":random_idx})
    return res['prompt']

async def add_one_qa_pair(qa_pair:QaPair):
    db = client["TACC_GPT"]
    qa_collection = db["QA"]
    res = await qa_collection.insert_one(qa_pair.dict())
    return res

async def add_one_ranking(rankingResults:RankingResults):
    db = client["TACC_GPT"]
    ranking_collection = db['ranking']
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
    db = client["TACC_GPT"]
    qa_collection = db["QA"]
    cursor = qa_collection.find({})
    qa_pairs = []
    async for qa_pair in cursor:
        qa_pairs.append(QaPair(**qa_pair))
    return qa_pairs

async def fetch_all_rankings():
    db = client["TACC_GPT"]
    ranking_collection = db['ranking']

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

async def main():
    JSON_FILE = "./prompt_data/first_prompts.json"
    db = client["TACC_GPT"]
    try:
        # if it exists, always drop all and populate it w/ prompts.json again
        first_prompts_collection = db['prompts']
        first_prompts_collection.remove()
    except:
        first_prompts_collection = db['prompts']

    with open(JSON_FILE,'r') as f:
        first_prompts = json.load(f)
    for idx, prompt in enumerate(first_prompts):
        res= await first_prompts_collection.insert_one({"_id":idx, "prompt":prompt}) 


if __name__ == '__main__':
    main()