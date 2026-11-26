"""Is the name a caller just gave the same person as a name on record for their number?

Code decides this, never a model. A wrong "yes" would read one person's notes to another, so a
doubtful match is a "no": the caller is then greeted as a new caller, which costs them little.

Two names match when no two different surnames are given ("Sam" and "Sam Carter" match, "Sam Pike"
and "Sam Carter" do not) and the first names match: they are equal after cleanup, when they are forms of one name (Bob and
Robert), or when they sound the same and start with the same letter (Steven and Stephen). Speech
to text often spells a name another way, so an exact match alone would miss real returning callers.
Known limit: some different names sound the same to the sound code (Dana and Diana).
"""

from __future__ import annotations

import unicodedata

import jellyfish

# Forms of one name. A name can be in more than one group (Sam: Samuel and Samantha).
NAME_GROUPS: tuple[frozenset[str], ...] = tuple(
    frozenset(group.split())
    for group in (
        "robert bob bobby rob robby robbie bert",
        "william bill billy will willy liam",
        "richard rick ricky rich richie dick",
        "james jim jimmy jamie",
        "john jon johnny jack",
        "jonathan jon jonny",
        "michael mike mikey mick mickey",
        "christopher chris kit topher",
        "matthew matt matty",
        "daniel dan danny",
        "david dave davey",
        "joseph joe joey",
        "thomas tom tommy",
        "anthony antony tony",
        "andrew andy drew",
        "steven stephen steve stevie",
        "edward ed eddie ted ned",
        "charles charlie chuck chas",
        "benjamin ben benny",
        "samuel sam sammy",
        "samantha sam sammy",
        "alexander alex al alec",
        "alexandra alex lexi alexa",
        "elizabeth liz lizzie beth betsy betty eliza",
        "katherine catherine kathryn katharine kate katie kathy cathy kat",
        "margaret maggie meg peggy",
        "jennifer jen jenny",
        "jessica jess jessie",
        "rebecca becky becca",
        "susan sue suzy susie",
        "patricia pat patty trish tricia",
        "patrick pat paddy",
        "deborah debra deb debbie",
        "nicholas nick nicky",
        "nicole nikki",
        "timothy tim timmy",
        "kenneth ken kenny",
        "ronald ron ronnie",
        "donald don donnie",
        "gregory greg",
        "jeffrey jeff geoffrey geoff",
        "joshua josh",
        "zachary zach zack",
        "nathaniel nathan nate",
        "abigail abby",
        "victoria vicky tori",
        "christine christina chris tina",
        "theodore theo ted teddy",
        "frederick fred freddie",
        "lawrence laurence larry",
        "gerald jerry",
        "raymond ray",
        "douglas doug",
        "peter pete",
        "philip phillip phil",
        "sean shaun shawn",
        "brian bryan",
        "ann anne annie",
        "sara sarah",
        "jacob jake",
        "kimberly kim",
        "amanda mandy",
        "michelle shelly",
    )
)


def clean_first_name(name: str) -> str:
    """The first word of a name, lower case, letters only, accents removed: "José-Luis Park" gives "joseluis"."""
    words = name.split()
    return _clean(words[0]) if words else ""


def _clean(word: str) -> str:
    plain = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode()
    return "".join(ch for ch in plain.casefold() if ch.isalpha())


def _surname(name: str) -> str:
    """The last word of a name of two or more words, cleaned. "" for a first name alone."""
    words = name.split()
    return _clean(words[-1]) if len(words) > 1 else ""


def same_person(spoken: str, stored: str) -> bool:
    """True when the names match by the rules above. An empty name never matches."""
    a, b = clean_first_name(spoken), clean_first_name(stored)
    if not a or not b:
        return False
    sa, sb = _surname(spoken), _surname(stored)
    if sa and sb and sa != sb:
        return False
    if a == b:
        return True
    if any(a in group and b in group for group in NAME_GROUPS):
        return True
    return len(a) >= 3 and len(b) >= 3 and a[0] == b[0] and jellyfish.metaphone(a) == jellyfish.metaphone(b)
