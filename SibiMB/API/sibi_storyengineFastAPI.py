from fastapi import FastAPI
from pydantic import BaseModel
import API.gptnarrativeAPI as sibi

app = FastAPI()

class storyworld(BaseModel):
    story: str
    characters: dict
    current_state: dict
    behavior_states: dict

class NPC(BaseModel):
    id: int
    personalitytype: str

# Endpoint for creating a new NPC
@app.post('/api/npc/create')
async def setCharacterPersonality(npc: NPC, personality: str):
    sibi.assignPersonalitytoCharacter(npc, personality)
    sibi.characters[npc.id] = npc
    return {"message": "NPC created", "npc": npc}

# Endpoint for updating an story world and get the next part of the story based on the characters and their personalities 
@app.post('/api/npc/update_story')
async def update_story(next_action: str, actionindex: int, eventlist: list, story: str, characters: dict):
    nextevent = sibi.update_story_data(next_action, actionindex=actionindex, eventlist=eventlist, story=story, characters=characters)
    story = story + nextevent
    return {"message": "story updated", "updated_story": story}

# Endpoint for getting behavior predictions for an NPC
@app.get('/api/npc/get_behavior')
async def get_next_behavior(story: str, characters: dict, current_state: dict, behavior_states: dict):
    next_action, actionindex, eventlist = sibi.determine_next_action(story=story, characters=characters, current_state=current_state, behavior_states=behavior_states)
    return {"behavior": next_action, "actionindex": actionindex, "eventlist": eventlist}


# Endpoint for getting the next image that describes the scene
@app.get('/api/npc/get_image')
async def get_image(nexteventimg: str, characters: dict):
    imageprompt = sibi.genSceneImage(nexteventimg, characters)
    img_data = sibi.genImagefromScene(imageprompt=imageprompt)
    return {"image": img_data}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
