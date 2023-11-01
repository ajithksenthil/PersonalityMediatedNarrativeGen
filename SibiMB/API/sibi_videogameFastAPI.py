from fastapi import FastAPI
from pydantic import BaseModel
import API.gptnarrativeAPI as sibi

app = FastAPI()

class NPC(BaseModel):
    id: int
    traits: dict
    past_actions: list
    possible_actions: list
    context: dict
    world_elements: dict
    description: dict
    personalitytype: str

# Endpoint for creating a new NPC
@app.post('/api/npc/create')
async def create_npc(npc: NPC):
    sibi.create_npc(npc)
    return {"message": "NPC created", "npc": npc}

# Endpoint for updating an NPC's personality traits
@app.post('/api/npc/update_traits')
async def update_traits(id: int, traits: dict):
    sibi.update_traits(id, traits)
    return {"message": "NPC traits updated"}

# Endpoint for updating an NPC's context and world elements
@app.post('/api/npc/update_context')
async def update_context(id: int, context: dict, world_elements: dict):
    sibi.update_context(id, context, world_elements)
    return {"message": "NPC context updated"}

# Endpoint for updating an NPC's past actions
@app.post('/api/npc/update_past_actions')
async def update_past_actions(id: int, past_actions: list):
    sibi.update_past_actions(id, past_actions)
    return {"message": "NPC past actions updated"}

# Endpoint for updating an NPC's possible actions
@app.post('/api/npc/update_possible_actions')
async def update_possible_actions(id: int, possible_actions: list):
    sibi.update_possible_actions(id, possible_actions)
    return {"message": "NPC possible actions updated"}

# Endpoint for getting behavior predictions for an NPC
@app.get('/api/npc/get_behavior')
async def get_behavior(id: int):
    behavior = sibi.get_behavior(id)
    return {"behavior": behavior}

# Additional endpoints for other actions

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
