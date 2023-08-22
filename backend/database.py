import motor.motor_asyncio
from models import QA_pair, RankingResults

client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://database:27017")
db = client["TACC_GPT_test"]
qa_collection = db["QA"]
ranking_collection = db['ranking']

async def add_one_qa_pair(qa_pair:QA_pair):
    res = await qa_collection.insert_one(qa_pair)
    return res

async def add_one_ranking(rankingResults:RankingResults):
    rankingResults_dict = {"prompt": rankingResults.prompt}
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
        qa_pairs.append(QA_pair(**qa_pair))
    return qa_pairs

async def fetch_all_rankings():
    cursor = ranking_collection.find({})
    rankings = []
    async for ranking in cursor:
        rankings.append(ranking)
    return rankings