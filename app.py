import asyncio
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google import genai
from google.oauth2 import service_account

from match_engine import MatchEngine

load_dotenv()

# Gemini client via Vertex AI with service account
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(__file__), "service-account.json")
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
gemini = genai.Client(
    vertexai=True,
    project="useful-maxim-496613-r8",
    location="us-central1",
    credentials=credentials,
)
GEMINI_MODEL = "gemini-2.5-flash"

# Global state
match: MatchEngine | None = None
match_task: asyncio.Task | None = None
clients: dict[WebSocket, dict] = {}  # ws -> {username, predictions, score, streak}
match_running = False
prediction_window_open = False
current_predictions: dict[str, str] = {}  # username -> prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup
    if match_task and not match_task.done():
        match_task.cancel()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# --- Helpers ---

async def broadcast(message: dict):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.pop(ws, None)


async def send_to(ws: WebSocket, message: dict):
    try:
        await ws.send_json(message)
    except Exception:
        pass


def get_leaderboard():
    board = []
    for ws, info in clients.items():
        board.append({
            "username": info["username"],
            "score": info["score"],
            "streak": info["streak"],
            "correct": info["correct"],
            "total": info["total"],
        })
    board.sort(key=lambda x: x["score"], reverse=True)
    return board[:20]


POINTS = {"0": 10, "1": 10, "2": 15, "3": 20, "4": 25, "6": 30, "W": 50}
STREAK_BONUS = {3: 10, 5: 25, 10: 50}

# Merchandise Store
MERCH_CATALOG = [
    {"id": "sticker", "name": "Team Sticker Pack", "price": 50, "emoji": "🏷️", "desc": "Set of 5 IPL team stickers"},
    {"id": "badge", "name": "Fan Badge", "price": 100, "emoji": "🎖️", "desc": "Digital fan badge for your profile"},
    {"id": "cap", "name": "Team Cap", "price": 200, "emoji": "🧢", "desc": "Official IPL team cap"},
    {"id": "jersey", "name": "Team Jersey", "price": 500, "emoji": "👕", "desc": "Official team jersey"},
    {"id": "bat", "name": "Signed Mini Bat", "price": 800, "emoji": "🏏", "desc": "Miniature bat with player signature"},
    {"id": "vip", "name": "VIP Match Pass", "price": 1500, "emoji": "🎫", "desc": "VIP access to next home game"},
]


# --- AI Cricket Agent (Function Calling) ---

def _tool_get_match_state() -> dict:
    """Get current match score, overs, batting/bowling teams."""
    if not match:
        return {"error": "No match in progress"}
    return match.get_state()


def _tool_get_batsman_stats(name: str = "") -> dict:
    """Get batting statistics for a specific batsman or all current batsmen."""
    if not match:
        return {"error": "No match in progress"}
    if name:
        stat = match.batsmen_stats.get(name)
        if stat:
            return stat.to_dict()
        # Fuzzy match
        for n, s in match.batsmen_stats.items():
            if name.lower() in n.lower():
                return s.to_dict()
        return {"error": f"Batsman '{name}' not found"}
    return {"batsmen": [s.to_dict() for s in match.batsmen_stats.values()]}


def _tool_get_bowler_stats(name: str = "") -> dict:
    """Get bowling statistics for a specific bowler or all bowlers."""
    if not match:
        return {"error": "No match in progress"}
    if name:
        stat = match.bowler_stats.get(name)
        if stat:
            return stat.to_dict()
        for n, s in match.bowler_stats.items():
            if name.lower() in n.lower():
                return s.to_dict()
        return {"error": f"Bowler '{name}' not found"}
    return {"bowlers": [s.to_dict() for s in match.bowler_stats.values()]}


def _tool_get_recent_balls(count: int = 6) -> list:
    """Get the last N ball events with descriptions."""
    if not match:
        return []
    recent = match.ball_history[-count:]
    return [b.to_dict() for b in recent]


def _tool_get_match_phase_info() -> dict:
    """Get strategic info: phase, run rate, required rate, balls remaining."""
    if not match:
        return {"error": "No match in progress"}
    state = match.get_state()
    return {
        "phase": state["phase"],
        "innings": state["innings"],
        "run_rate": state["run_rate"],
        "required_rate": state["required_rate"],
        "target": state["target"],
        "balls_bowled": state["legal_balls"],
        "balls_remaining": 120 - state["legal_balls"],
        "wickets_in_hand": 10 - state["wickets"],
    }


def _tool_get_partnership_info() -> dict:
    """Get current partnership details between striker and non-striker."""
    if not match or match.is_complete:
        return {"error": "No active partnership"}
    striker = match.batsmen_stats.get(match.striker)
    non_striker = match.batsmen_stats.get(match.non_striker)
    if not striker or not non_striker:
        return {"error": "Partnership data unavailable"}
    return {
        "striker": striker.to_dict(),
        "non_striker": non_striker.to_dict(),
        "combined_runs": striker.runs + non_striker.runs,
        "combined_balls": striker.balls + non_striker.balls,
        "combined_boundaries": striker.fours + non_striker.fours + striker.sixes + non_striker.sixes,
    }


def _tool_get_innings1_scorecard() -> dict:
    """Get the first innings scorecard (only available during 2nd innings)."""
    if not match:
        return {"error": "No match in progress"}
    if match.innings1_scorecard:
        return match.innings1_scorecard
    return {"error": "First innings not complete yet"}


def _tool_get_prediction_stats(username: str) -> dict:
    """Get a user's prediction accuracy and patterns."""
    for ws, info in clients.items():
        if info["username"] == username:
            return {
                "username": username,
                "total_predictions": info["total"],
                "correct_predictions": info["correct"],
                "accuracy": round(info["correct"] / info["total"] * 100, 1) if info["total"] > 0 else 0,
                "current_streak": info["streak"],
                "score": info["score"],
                "prediction_history": info.get("prediction_history", []),
            }
    return {"error": f"User '{username}' not found"}


# Define tools for Gemini function calling
AGENT_TOOLS = [
    genai.types.Tool(function_declarations=[
        genai.types.FunctionDeclaration(
            name="get_match_state",
            description="Get current match score, overs, batting and bowling teams, and overall match status",
            parameters=genai.types.Schema(type="OBJECT", properties={}, required=[]),
        ),
        genai.types.FunctionDeclaration(
            name="get_batsman_stats",
            description="Get batting statistics (runs, balls, fours, sixes, strike rate) for a specific batsman by name, or all batsmen if name is empty",
            parameters=genai.types.Schema(
                type="OBJECT",
                properties={"name": genai.types.Schema(type="STRING", description="Batsman name (partial match supported). Leave empty for all batsmen.")},
                required=[],
            ),
        ),
        genai.types.FunctionDeclaration(
            name="get_bowler_stats",
            description="Get bowling statistics (overs, runs, wickets, economy) for a specific bowler by name, or all bowlers if name is empty",
            parameters=genai.types.Schema(
                type="OBJECT",
                properties={"name": genai.types.Schema(type="STRING", description="Bowler name (partial match supported). Leave empty for all bowlers.")},
                required=[],
            ),
        ),
        genai.types.FunctionDeclaration(
            name="get_recent_balls",
            description="Get details of the last N balls bowled including runs, wickets, and descriptions",
            parameters=genai.types.Schema(
                type="OBJECT",
                properties={"count": genai.types.Schema(type="INTEGER", description="Number of recent balls to retrieve (default 6)")},
                required=[],
            ),
        ),
        genai.types.FunctionDeclaration(
            name="get_match_phase_info",
            description="Get strategic match information including phase (powerplay/middle/death), run rate, required rate, balls remaining, wickets in hand",
            parameters=genai.types.Schema(type="OBJECT", properties={}, required=[]),
        ),
        genai.types.FunctionDeclaration(
            name="get_partnership_info",
            description="Get current batting partnership details - both batsmen stats and combined performance",
            parameters=genai.types.Schema(type="OBJECT", properties={}, required=[]),
        ),
        genai.types.FunctionDeclaration(
            name="get_innings1_scorecard",
            description="Get the first innings full scorecard (available only during second innings)",
            parameters=genai.types.Schema(type="OBJECT", properties={}, required=[]),
        ),
        genai.types.FunctionDeclaration(
            name="get_prediction_stats",
            description="Get a fan's prediction accuracy, streak, score, and prediction patterns",
            parameters=genai.types.Schema(
                type="OBJECT",
                properties={"username": genai.types.Schema(type="STRING", description="The fan's username")},
                required=["username"],
            ),
        ),
    ])
]

TOOL_FUNCTIONS = {
    "get_match_state": lambda args: _tool_get_match_state(),
    "get_batsman_stats": lambda args: _tool_get_batsman_stats(args.get("name", "")),
    "get_bowler_stats": lambda args: _tool_get_bowler_stats(args.get("name", "")),
    "get_recent_balls": lambda args: _tool_get_recent_balls(args.get("count", 6)),
    "get_match_phase_info": lambda args: _tool_get_match_phase_info(),
    "get_partnership_info": lambda args: _tool_get_partnership_info(),
    "get_innings1_scorecard": lambda args: _tool_get_innings1_scorecard(),
    "get_prediction_stats": lambda args: _tool_get_prediction_stats(args.get("username", "")),
}

AGENT_SYSTEM = (
    "You are an expert IPL cricket analyst agent. You have access to live match data tools. "
    "When a fan asks a question, use the tools to get real data before answering. "
    "Always call at least one tool to ground your answer in real match data. "
    "Be enthusiastic, use cricket terminology, and keep responses concise (2-4 sentences). "
    "You can analyze strategies, compare players, suggest what might happen next, and give tactical advice. "
    "When asked about predictions, use get_prediction_stats to analyze the fan's patterns."
)


async def run_cricket_agent(user_message: str, username: str) -> tuple[str, list[str]]:
    """Run the cricket agent with function calling. Returns (response, tools_used)."""
    tools_used = []
    try:
        contents = [
            genai.types.Content(role="user", parts=[
                genai.types.Part.from_text(text=f"[Fan: {username}] {user_message}")
            ])
        ]

        # Initial call with tools
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=AGENT_SYSTEM,
                tools=AGENT_TOOLS,
                temperature=0.7,
            ),
        )

        # Agentic loop - handle tool calls
        max_turns = 5
        for _ in range(max_turns):
            # Check if there are function calls
            func_calls = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    func_calls.append(part.function_call)

            if not func_calls:
                break

            # Execute function calls
            func_responses = []
            for fc in func_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}
                tools_used.append(fn_name)

                if fn_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[fn_name](fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                func_responses.append(
                    genai.types.Part.from_function_response(
                        name=fn_name,
                        response=result,
                    )
                )

            # Add assistant response and tool results to contents
            contents.append(response.candidates[0].content)
            contents.append(genai.types.Content(role="user", parts=func_responses))

            # Call again with tool results
            response = await gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=AGENT_SYSTEM,
                    tools=AGENT_TOOLS,
                    temperature=0.7,
                ),
            )

        # Extract final text
        final_text = ""
        for part in response.candidates[0].content.parts:
            if part.text:
                final_text += part.text

        return final_text.strip(), tools_used

    except Exception as e:
        return f"Sorry, I couldn't process that right now. Error: {str(e)[:100]}", []


# --- AI Prediction Coach ---

async def get_prediction_coaching(username: str, info: dict) -> str | None:
    """Generate personalized prediction coaching based on user's history."""
    if info["total"] < 5:
        return None  # Not enough data yet

    history = info.get("prediction_history", [])
    recent = history[-20:] if history else []

    try:
        prompt = (
            f"You are an IPL prediction coach. Analyze this fan's prediction data and give ONE short personalized tip.\n\n"
            f"Fan: {username}\n"
            f"Total predictions: {info['total']}\n"
            f"Correct: {info['correct']} ({round(info['correct']/info['total']*100, 1)}% accuracy)\n"
            f"Current streak: {info['streak']}\n"
            f"Recent predictions (newest first): {json.dumps(recent[-10:])}\n\n"
            f"Give a specific, actionable tip in 1-2 sentences. Reference their actual patterns. "
            f"Examples: 'You've predicted 4 in the last 3 balls but none hit — the bowler is bowling tight, try dot or single.' "
            f"or 'Great read on that wicket! This bowler has been getting movement, keep backing W.'"
        )
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception:
        return None


async def get_ai_commentary(ball_event, match_state):
    """Get Gemini AI commentary for a ball event."""
    try:
        ctx = (
            f"IPL Cricket Match: {match_state['batting_team']} vs {match_state['bowling_team']}. "
            f"Score: {match_state['score']}/{match_state['wickets']} in {match_state['overs']} overs. "
            f"{'Target: ' + str(match_state['target']) + '. ' if match_state['target'] else ''}"
            f"Phase: {match_state['phase']}. "
        )
        ball = ball_event.to_dict()
        prompt = (
            f"{ctx}"
            f"Ball event: {ball['description']}. "
            f"{'This is a KEY MOMENT: ' + ball['key_moment_type'] + '!' if ball['is_key_moment'] else ''}\n\n"
            f"Give a short, exciting TV-style cricket commentary line (1-2 sentences max). "
            f"Be enthusiastic for boundaries/wickets. Use cricket terminology. Keep it punchy."
        )
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return ball_event.description


async def get_ai_trivia():
    """Generate a cricket trivia question using Gemini."""
    try:
        prompt = (
            "Generate a fun IPL cricket trivia question with 4 options. "
            "Return ONLY valid JSON in this format (no markdown):\n"
            '{"question": "...", "options": ["A", "B", "C", "D"], "correct": 0, "fact": "short fun fact"}\n'
            "Make it about IPL records, famous moments, or legendary players. "
            "The correct index should be random (0-3)."
        )
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        text = response.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return {
            "question": "Who has the most IPL centuries?",
            "options": ["Virat Kohli", "Chris Gayle", "Rohit Sharma", "David Warner"],
            "correct": 0,
            "fact": "Virat Kohli holds the record for most centuries in IPL history!"
        }


async def get_ai_insight(match_state):
    """Get a strategic insight about the match."""
    try:
        prompt = (
            f"IPL Match: {match_state['batting_team']} {match_state['score']}/{match_state['wickets']} "
            f"in {match_state['overs']} overs. "
            f"{'Chasing ' + str(match_state['target']) + '. Required rate: ' + str(match_state['required_rate']) + '. ' if match_state['target'] else ''}"
            f"Current run rate: {match_state['run_rate']}. Phase: {match_state['phase']}.\n\n"
            f"Give ONE short strategic insight or stat (1-2 sentences). "
            f"Example: 'CSK need 48 off 24 - historically teams win 35% of such chases in IPL.' "
            f"Be specific and data-oriented."
        )
        response = await gemini.aio.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text.strip()
    except Exception:
        return None


# --- Match Loop ---

async def match_loop():
    global match, match_running, prediction_window_open, current_predictions

    match_running = True
    ball_count = 0

    await broadcast({"type": "match_start", "state": match.get_state()})
    await asyncio.sleep(2)

    while not match.is_complete and match_running:
        # Open prediction window
        prediction_window_open = True
        current_predictions = {}
        await broadcast({
            "type": "prediction_open",
            "countdown": 6,
            "state": match.get_state()
        })
        await asyncio.sleep(6)

        # Close prediction window
        prediction_window_open = False
        await broadcast({"type": "prediction_closed"})
        await asyncio.sleep(0.5)

        # Simulate ball
        event = match.simulate_ball()
        if event is None:
            break

        state = match.get_state()
        ball_count += 1

        # Get AI commentary (don't block on it - use timeout)
        try:
            commentary = await asyncio.wait_for(
                get_ai_commentary(event, state), timeout=4.0
            )
        except asyncio.TimeoutError:
            commentary = event.description

        # Score predictions
        outcome_label = event.to_dict()["outcome_label"]
        for ws, info in clients.items():
            username = info["username"]
            if username in current_predictions:
                info["total"] += 1
                info["prediction_history"].append({
                    "predicted": current_predictions[username],
                    "actual": outcome_label,
                    "correct": current_predictions[username] == outcome_label,
                    "over": event.overs_display,
                })
                if current_predictions[username] == outcome_label:
                    pts = POINTS.get(outcome_label, 10)
                    info["streak"] += 1
                    # Streak bonus
                    for threshold, bonus in STREAK_BONUS.items():
                        if info["streak"] == threshold:
                            pts += bonus
                    info["score"] += pts
                    info["correct"] += 1
                    await send_to(ws, {
                        "type": "prediction_result",
                        "correct": True,
                        "points": pts,
                        "streak": info["streak"],
                        "total_score": info["score"]
                    })
                else:
                    info["streak"] = 0
                    await send_to(ws, {
                        "type": "prediction_result",
                        "correct": False,
                        "points": 0,
                        "streak": 0,
                        "total_score": info["score"]
                    })

        # Broadcast ball result
        await broadcast({
            "type": "ball_result",
            "ball": event.to_dict(),
            "commentary": commentary,
            "state": state,
            "leaderboard": get_leaderboard()
        })

        # AI insight every 12 balls or on key moments
        if ball_count % 12 == 0 or event.is_key_moment:
            insight = await get_ai_insight(state)
            if insight:
                await broadcast({"type": "ai_insight", "insight": insight})

        # Trivia at end of each over (every 6 legal balls)
        if state["legal_balls"] > 0 and state["legal_balls"] % 6 == 0 and not match.is_complete:
            trivia = await get_ai_trivia()
            await broadcast({"type": "trivia", "trivia": trivia})

        # Prediction coaching every 8 balls
        if ball_count % 8 == 0:
            for ws, info in clients.items():
                coaching = await get_prediction_coaching(info["username"], info)
                if coaching:
                    await send_to(ws, {"type": "coaching_tip", "tip": coaching})

        # Check innings change
        if match.innings == 2 and state.get("innings") == 1:
            await broadcast({
                "type": "innings_break",
                "innings1": match.innings1_scorecard,
                "target": match.target,
                "state": match.get_state()
            })
            await asyncio.sleep(3)

        # Pause between balls
        await asyncio.sleep(2)

    # Match complete
    if match.is_complete:
        await broadcast({
            "type": "match_complete",
            "result": match.result,
            "state": match.get_state(),
            "leaderboard": get_leaderboard()
        })

    match_running = False


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global match, match_task, match_running

    await websocket.accept()
    clients[websocket] = {
        "username": "Fan",
        "score": 0,
        "streak": 0,
        "correct": 0,
        "total": 0,
        "inventory": [],
        "prediction_history": [],
    }

    try:
        # Send current state
        if match:
            await send_to(websocket, {
                "type": "state_update",
                "state": match.get_state(),
                "leaderboard": get_leaderboard(),
                "match_running": match_running
            })
        else:
            await send_to(websocket, {"type": "waiting"})

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "set_username":
                clients[websocket]["username"] = data["username"]
                await broadcast({"type": "leaderboard", "leaderboard": get_leaderboard()})

            elif msg_type == "start_match":
                if not match_running:
                    match = MatchEngine()
                    match_task = asyncio.create_task(match_loop())

            elif msg_type == "prediction":
                if prediction_window_open:
                    username = clients[websocket]["username"]
                    current_predictions[username] = data["value"]
                    await send_to(websocket, {"type": "prediction_ack", "value": data["value"]})

            elif msg_type == "reaction":
                await broadcast({
                    "type": "reaction",
                    "username": clients[websocket]["username"],
                    "emoji": data["emoji"]
                })

            elif msg_type == "get_merch":
                await send_to(websocket, {
                    "type": "merch_catalog",
                    "items": MERCH_CATALOG,
                    "balance": clients[websocket]["score"],
                    "inventory": clients[websocket]["inventory"],
                })

            elif msg_type == "buy_merch":
                item_id = data.get("item_id")
                item = next((m for m in MERCH_CATALOG if m["id"] == item_id), None)
                if item and clients[websocket]["score"] >= item["price"]:
                    clients[websocket]["score"] -= item["price"]
                    clients[websocket]["inventory"].append(item_id)
                    await send_to(websocket, {
                        "type": "merch_bought",
                        "success": True,
                        "item": item,
                        "balance": clients[websocket]["score"],
                        "inventory": clients[websocket]["inventory"],
                    })
                    await broadcast({"type": "leaderboard", "leaderboard": get_leaderboard()})
                    # Announce purchase
                    await broadcast({
                        "type": "merch_announce",
                        "username": clients[websocket]["username"],
                        "item_name": item["name"],
                        "item_emoji": item["emoji"],
                    })
                else:
                    await send_to(websocket, {
                        "type": "merch_bought",
                        "success": False,
                        "message": "Not enough credits!" if item else "Item not found",
                        "balance": clients[websocket]["score"],
                    })

            elif msg_type == "agent_chat":
                user_msg = data.get("message", "")
                uname = clients[websocket]["username"]
                await send_to(websocket, {"type": "agent_typing"})
                try:
                    reply, tools_used = await asyncio.wait_for(
                        run_cricket_agent(user_msg, uname), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    reply = "Sorry, took too long to analyze. Try a simpler question!"
                    tools_used = []
                await send_to(websocket, {
                    "type": "agent_reply",
                    "message": reply,
                    "tools_used": tools_used,
                })

            elif msg_type == "trivia_answer":
                # Points for correct trivia
                if data.get("correct"):
                    clients[websocket]["score"] += 15
                    await send_to(websocket, {
                        "type": "trivia_result",
                        "correct": True,
                        "points": 15,
                        "total_score": clients[websocket]["score"]
                    })
                else:
                    await send_to(websocket, {
                        "type": "trivia_result",
                        "correct": False,
                        "points": 0,
                        "total_score": clients[websocket]["score"]
                    })
                await broadcast({"type": "leaderboard", "leaderboard": get_leaderboard()})

    except WebSocketDisconnect:
        clients.pop(websocket, None)
        await broadcast({"type": "leaderboard", "leaderboard": get_leaderboard()})
    except Exception:
        clients.pop(websocket, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
