import random
from dataclasses import dataclass, field
from typing import Optional


# IPL Teams with real player names
TEAMS = {
    "CSK": {
        "name": "Chennai Super Kings",
        "short": "CSK",
        "color": "#FFCB05",
        "batting": [
            "Ruturaj Gaikwad", "Devon Conway", "Ajinkya Rahane",
            "Shivam Dube", "Ravindra Jadeja", "MS Dhoni",
            "Moeen Ali", "Deepak Chahar", "Shardul Thakur",
            "Tushar Deshpande", "Matheesha Pathirana"
        ],
        "bowling": [
            "Deepak Chahar", "Matheesha Pathirana", "Tushar Deshpande",
            "Shardul Thakur", "Ravindra Jadeja", "Moeen Ali"
        ]
    },
    "MI": {
        "name": "Mumbai Indians",
        "short": "MI",
        "color": "#004BA0",
        "batting": [
            "Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav",
            "Tilak Varma", "Hardik Pandya", "Tim David",
            "Nehal Wadhera", "Jasprit Bumrah", "Piyush Chawla",
            "Akash Madhwal", "Gerald Coetzee"
        ],
        "bowling": [
            "Jasprit Bumrah", "Gerald Coetzee", "Akash Madhwal",
            "Piyush Chawla", "Hardik Pandya", "Tilak Varma"
        ]
    }
}

# Ball outcome probabilities by phase
PROBABILITIES = {
    "powerplay": {  # Overs 1-6
        0: 0.33, 1: 0.28, 2: 0.07, 3: 0.01, 4: 0.15, 6: 0.09, -1: 0.05, -2: 0.02
    },
    "middle": {  # Overs 7-15
        0: 0.36, 1: 0.30, 2: 0.09, 3: 0.01, 4: 0.11, 6: 0.06, -1: 0.05, -2: 0.02
    },
    "death": {  # Overs 16-20
        0: 0.22, 1: 0.22, 2: 0.07, 3: 0.02, 4: 0.18, 6: 0.16, -1: 0.10, -2: 0.03
    }
}

DISMISSAL_TYPES = [
    "Caught", "Bowled", "LBW", "Run Out", "Stumped", "Caught & Bowled"
]


@dataclass
class BatsmanStats:
    name: str
    runs: int = 0
    balls: int = 0
    fours: int = 0
    sixes: int = 0
    is_out: bool = False
    how_out: str = ""

    @property
    def strike_rate(self):
        return round((self.runs / self.balls) * 100, 1) if self.balls > 0 else 0.0

    def to_dict(self):
        return {
            "name": self.name, "runs": self.runs, "balls": self.balls,
            "fours": self.fours, "sixes": self.sixes, "is_out": self.is_out,
            "how_out": self.how_out, "strike_rate": self.strike_rate
        }


@dataclass
class BowlerStats:
    name: str
    overs: int = 0  # balls bowled (legal)
    runs: int = 0
    wickets: int = 0
    extras: int = 0

    @property
    def overs_display(self):
        return f"{self.overs // 6}.{self.overs % 6}"

    @property
    def economy(self):
        overs = self.overs / 6
        return round(self.runs / overs, 1) if overs > 0 else 0.0

    def to_dict(self):
        return {
            "name": self.name, "overs": self.overs_display,
            "runs": self.runs, "wickets": self.wickets, "economy": self.economy
        }


@dataclass
class BallEvent:
    over: int
    ball_in_over: int
    runs: int
    is_wicket: bool
    is_boundary: bool
    is_six: bool
    is_wide: bool
    batsman: str
    bowler: str
    description: str
    total_score: int
    total_wickets: int
    overs_display: str
    is_key_moment: bool = False
    key_moment_type: str = ""

    def to_dict(self):
        return {
            "over": self.over, "ball_in_over": self.ball_in_over,
            "runs": self.runs, "is_wicket": self.is_wicket,
            "is_boundary": self.is_boundary, "is_six": self.is_six,
            "is_wide": self.is_wide, "batsman": self.batsman,
            "bowler": self.bowler, "description": self.description,
            "total_score": self.total_score, "total_wickets": self.total_wickets,
            "overs_display": self.overs_display,
            "is_key_moment": self.is_key_moment,
            "key_moment_type": self.key_moment_type,
            "outcome_label": self._outcome_label()
        }

    def _outcome_label(self):
        if self.is_wicket:
            return "W"
        if self.is_wide:
            return "WD"
        return str(self.runs)


class MatchEngine:
    def __init__(self, team1_key="CSK", team2_key="MI"):
        self.team1 = TEAMS[team1_key]
        self.team2 = TEAMS[team2_key]
        self.innings = 1
        self.score = 0
        self.wickets = 0
        self.legal_balls = 0  # total legal balls bowled
        self.balls_in_over = 0  # legal balls in current over
        self.current_over = 0
        self.target: Optional[int] = None
        self.is_complete = False
        self.result = ""

        # Batting
        self.batting_team = self.team1
        self.bowling_team = self.team2
        self.batting_order = list(self.batting_team["batting"])
        self.striker_idx = 0
        self.non_striker_idx = 1
        self.next_batsman_idx = 2
        self.batsmen_stats: dict[str, BatsmanStats] = {}
        self._init_batsman(self.batting_order[0])
        self._init_batsman(self.batting_order[1])

        # Bowling
        self.bowling_order = list(self.bowling_team["bowling"])
        self.current_bowler_idx = 0
        self.bowler_stats: dict[str, BowlerStats] = {}
        self._init_bowler(self.bowling_order[0])

        # History
        self.ball_history: list[BallEvent] = []
        self.over_history: list[list[str]] = [[]]  # current over balls display
        self.this_over: list[str] = []

        # Innings 1 storage
        self.innings1_scorecard = None

    def _init_batsman(self, name):
        if name not in self.batsmen_stats:
            self.batsmen_stats[name] = BatsmanStats(name=name)

    def _init_bowler(self, name):
        if name not in self.bowler_stats:
            self.bowler_stats[name] = BowlerStats(name=name)

    @property
    def striker(self):
        return self.batting_order[self.striker_idx]

    @property
    def non_striker(self):
        return self.batting_order[self.non_striker_idx]

    @property
    def current_bowler(self):
        return self.bowling_order[self.current_bowler_idx]

    @property
    def overs_display(self):
        return f"{self.legal_balls // 6}.{self.legal_balls % 6}"

    @property
    def phase(self):
        over = self.legal_balls // 6
        if over < 6:
            return "powerplay"
        elif over < 15:
            return "middle"
        else:
            return "death"

    @property
    def run_rate(self):
        overs = self.legal_balls / 6
        return round(self.score / overs, 2) if overs > 0 else 0.0

    @property
    def required_rate(self):
        if self.target is None:
            return None
        remaining_runs = self.target - self.score
        remaining_balls = 120 - self.legal_balls
        remaining_overs = remaining_balls / 6
        if remaining_overs <= 0:
            return None
        return round(remaining_runs / remaining_overs, 2)

    def simulate_ball(self) -> BallEvent:
        if self.is_complete:
            return None

        probs = PROBABILITIES[self.phase]

        # Adjust if chasing
        if self.target and self.innings == 2:
            needed = self.target - self.score
            remaining = 120 - self.legal_balls
            if remaining > 0 and needed / (remaining / 6) > 12:
                # Under pressure - more aggressive + more wickets
                probs = dict(probs)
                probs[4] *= 1.3
                probs[6] *= 1.4
                probs[-1] *= 1.3
                total = sum(probs.values())
                probs = {k: v / total for k, v in probs.items()}

        outcomes = list(probs.keys())
        weights = list(probs.values())
        outcome = random.choices(outcomes, weights=weights, k=1)[0]

        is_wide = outcome == -2
        is_wicket = outcome == -1
        is_boundary = outcome == 4
        is_six = outcome == 6
        runs = 0 if is_wicket else (1 if is_wide else outcome)

        batsman_name = self.striker
        bowler_name = self.current_bowler
        batsman = self.batsmen_stats[batsman_name]
        bowler = self.bowler_stats[bowler_name]

        # Update stats
        if is_wide:
            self.score += 1
            bowler.runs += 1
            bowler.extras += 1
            desc = f"Wide ball! 1 extra run"
        elif is_wicket:
            dismissal = random.choice(DISMISSAL_TYPES)
            batsman.is_out = True
            batsman.balls += 1
            batsman.how_out = f"{dismissal} b {bowler_name}"
            bowler.overs += 1
            bowler.wickets += 1
            self.wickets += 1
            self.balls_in_over += 1
            self.legal_balls += 1
            desc = f"OUT! {batsman_name} {dismissal.lower()} by {bowler_name} for {batsman.runs}({batsman.balls})"
        else:
            batsman.runs += runs
            batsman.balls += 1
            if is_boundary:
                batsman.fours += 1
            if is_six:
                batsman.sixes += 1
            self.score += runs
            bowler.runs += runs
            bowler.overs += 1
            self.balls_in_over += 1
            self.legal_balls += 1

            if is_six:
                desc = f"SIX! {batsman_name} smashes it out of the park!"
            elif is_boundary:
                desc = f"FOUR! {batsman_name} finds the boundary!"
            elif runs == 0:
                desc = f"Dot ball. Good delivery by {bowler_name}"
            else:
                desc = f"{runs} run{'s' if runs > 1 else ''} taken by {batsman_name}"

        # Determine key moments
        is_key = False
        key_type = ""
        if is_wicket:
            is_key = True
            key_type = "wicket"
        elif is_six:
            is_key = True
            key_type = "six"
        elif batsman.runs in [50, 100] and not is_wicket:
            is_key = True
            key_type = f"milestone_{batsman.runs}"
        elif self.target and self.score >= self.target:
            is_key = True
            key_type = "chase_complete"

        # This over display
        if is_wide:
            self.this_over.append("WD")
        elif is_wicket:
            self.this_over.append("W")
        else:
            label = str(runs)
            if is_boundary:
                label = "4"
            if is_six:
                label = "6"
            self.this_over.append(label)

        event = BallEvent(
            over=self.current_over,
            ball_in_over=self.balls_in_over,
            runs=runs if not is_wide else 0,
            is_wicket=is_wicket,
            is_boundary=is_boundary,
            is_six=is_six,
            is_wide=is_wide,
            batsman=batsman_name,
            bowler=bowler_name,
            description=desc,
            total_score=self.score,
            total_wickets=self.wickets,
            overs_display=self.overs_display,
            is_key_moment=is_key,
            key_moment_type=key_type
        )
        self.ball_history.append(event)

        # Rotate strike
        if not is_wide:
            if is_wicket:
                if self.wickets < 10 and self.next_batsman_idx < len(self.batting_order):
                    self.striker_idx = self.next_batsman_idx
                    self._init_batsman(self.batting_order[self.striker_idx])
                    self.next_batsman_idx += 1
            elif runs % 2 == 1:
                self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx

        # End of over
        if not is_wide and self.balls_in_over >= 6:
            self.current_over += 1
            self.balls_in_over = 0
            self.over_history.append(list(self.this_over))
            self.this_over = []
            # Switch strike at end of over
            self.striker_idx, self.non_striker_idx = self.non_striker_idx, self.striker_idx
            # Change bowler
            self.current_bowler_idx = (self.current_bowler_idx + 1) % len(self.bowling_order)
            self._init_bowler(self.current_bowler)

        # Check innings/match end
        if self.legal_balls >= 120 or self.wickets >= 10:
            self._end_innings()
        elif self.target and self.score >= self.target:
            self._end_match_chase()

        return event

    def _end_innings(self):
        if self.innings == 1:
            self.innings1_scorecard = self._build_scorecard()
            self.target = self.score + 1
            # Reset for innings 2
            self.innings = 2
            self.score = 0
            self.wickets = 0
            self.legal_balls = 0
            self.balls_in_over = 0
            self.current_over = 0
            self.this_over = []
            self.over_history = [[]]

            # Swap teams
            self.batting_team, self.bowling_team = self.bowling_team, self.batting_team
            self.batting_order = list(self.batting_team["batting"])
            self.bowling_order = list(self.bowling_team["bowling"])
            self.striker_idx = 0
            self.non_striker_idx = 1
            self.next_batsman_idx = 2
            self.batsmen_stats = {}
            self._init_batsman(self.batting_order[0])
            self._init_batsman(self.batting_order[1])
            self.bowler_stats = {}
            self.current_bowler_idx = 0
            self._init_bowler(self.bowling_order[0])
        else:
            # Match over
            self.is_complete = True
            chasing = self.batting_team["short"]
            bowling = self.bowling_team["short"]
            if self.score >= self.target:
                self.result = f"{chasing} won by {10 - self.wickets} wickets!"
            else:
                self.result = f"{bowling} won by {self.target - self.score - 1} runs!"

    def _end_match_chase(self):
        self.is_complete = True
        chasing = self.batting_team["short"]
        self.result = f"{chasing} won by {10 - self.wickets} wickets with {120 - self.legal_balls} balls remaining!"

    def _build_scorecard(self):
        return {
            "team": self.batting_team["short"],
            "team_name": self.batting_team["name"],
            "score": self.score,
            "wickets": self.wickets,
            "overs": self.overs_display,
            "batsmen": [s.to_dict() for s in self.batsmen_stats.values()],
            "bowlers": [s.to_dict() for s in self.bowler_stats.values()],
        }

    def get_state(self):
        return {
            "innings": self.innings,
            "batting_team": self.batting_team["short"],
            "batting_team_name": self.batting_team["name"],
            "batting_team_color": self.batting_team["color"],
            "bowling_team": self.bowling_team["short"],
            "bowling_team_name": self.bowling_team["name"],
            "score": self.score,
            "wickets": self.wickets,
            "overs": self.overs_display,
            "legal_balls": self.legal_balls,
            "target": self.target,
            "run_rate": self.run_rate,
            "required_rate": self.required_rate,
            "striker": self.batsmen_stats.get(self.striker, BatsmanStats(self.striker)).to_dict() if not self.is_complete else None,
            "non_striker": self.batsmen_stats.get(self.non_striker, BatsmanStats(self.non_striker)).to_dict() if not self.is_complete else None,
            "bowler": self.bowler_stats.get(self.current_bowler, BowlerStats(self.current_bowler)).to_dict() if not self.is_complete else None,
            "this_over": self.this_over,
            "is_complete": self.is_complete,
            "result": self.result,
            "phase": self.phase,
            "innings1": self.innings1_scorecard,
            "batsmen": [s.to_dict() for s in self.batsmen_stats.values()],
        }
