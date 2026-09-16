"""
Week 3 Day 3 — AFL Domain Vector Store & Semantic Retrieval
===========================================================
Implements a clean LangChain VectorStore over unstructured AFL domain text:
- AFL Rules, scoring, and terminology (behind, mark, disposal, holding the ball)
- AFL Clubs, heritage, and key personnel
- AFL Stadiums and marquee venues (MCG, Marvel Stadium, Gabba, Adelaide Oval, Optus Stadium, SCG)
- AFL Awards and honors (Brownlow Medal, Coleman Medal, Norm Smith Medal, Premierships)
"""

from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# Comprehensive AFL Domain Knowledge Base Documents
AFL_KNOWLEDGE_DOCS = [
    # Rules & Scoring
    Document(
        page_content=(
            "AFL Scoring Rules: A goal is awarded when the football is kicked cleanly between the two tall central "
            "goal posts without being touched by another player or hitting a post. A goal is worth 6 points. "
            "A behind is awarded when the ball passes between a tall goal post and a shorter outer behind post, "
            "hits a goal post, or is carried or touched over the goal line. A behind is worth 1 point. "
            "Final scores are reported as (Goals.Behinds Total_Points), for example 12.14 (86)."
        ),
        metadata={"category": "rules", "topic": "scoring", "title": "AFL Goals and Behinds Scoring"}
    ),
    Document(
        page_content=(
            "AFL Mark Rule: A mark is credited when a player catches the football cleanly from a kick delivered "
            "by another player that has traveled at least 15 meters in the air without touching the ground or being "
            "touched by another player in flight. Upon taking a mark, the player is awarded an unimpeded free kick "
            "from behind the spot of the mark."
        ),
        metadata={"category": "rules", "topic": "mark", "title": "AFL Mark and Contested Marks"}
    ),
    Document(
        page_content=(
            "AFL Disposals and Handling Rules: A disposal is defined as legally transferring possession of the ball "
            "via a kick or a handpass. A handpass is performed by holding the ball in one hand and tapping or punching "
            "it cleanly with the clenched fist of the opposite hand. Throwing, dropping, or pushing the ball is an "
            "illegal disposal and results in a free kick to the opposition."
        ),
        metadata={"category": "rules", "topic": "disposal", "title": "AFL Disposals: Kicks and Handpasses"}
    ),
    Document(
        page_content=(
            "Holding the Ball Rule: A tackling player is rewarded a free kick for holding the ball under three circumstances: "
            "1) Prior Opportunity: The ball-carrier had a reasonable chance to dispose of the ball before being tackled and "
            "failed to legally kick or handpass. 2) No Prior Opportunity: If the ball-carrier had no prior opportunity, they "
            "must make a genuine attempt to dispose of the ball when tackled legally. 3) Diving on the ball: A player who dives "
            "on the ball or drags it underneath themselves must immediately knock it out."
        ),
        metadata={"category": "rules", "topic": "tackling", "title": "Holding the Ball and Prior Opportunity"}
    ),
    Document(
        page_content=(
            "AFL 50-Metre Penalty Rule: A 50-meter penalty advances the attacking team 50 meters closer to their goal. "
            "It is awarded when an opponent infringes after a mark or free kick is paid, including encroaching the "
            "protected 10-meter area, wasting time by refusing to return the ball on the full, or engaging in late contact."
        ),
        metadata={"category": "rules", "topic": "penalties", "title": "AFL 50-Metre Penalty Rule"}
    ),

    # Stadiums & Grounds
    Document(
        page_content=(
            "Melbourne Cricket Ground (MCG): Located in Yarra Park, Melbourne, the MCG is the spiritual home of Australian "
            "rules football with an official capacity of 100,024 spectators. It hosts the annual AFL Grand Final, "
            "the Anzac Day blockbuster between Collingwood and Essendon, and serves as the primary home ground for "
            "Collingwood, Richmond, Hawthorn, and Melbourne."
        ),
        metadata={"category": "venues", "topic": "mcg", "title": "Melbourne Cricket Ground (MCG)"}
    ),
    Document(
        page_content=(
            "Marvel Stadium (Docklands Stadium): Located in the Docklands precinct of Melbourne, Marvel Stadium has a "
            "seating capacity of 53,359 and features a retractable roof that guarantees dry conditions. It is the primary "
            "home ground for St Kilda, Western Bulldogs, Essendon, and Carlton, and hosts North Melbourne fixtures."
        ),
        metadata={"category": "venues", "topic": "marvel", "title": "Marvel Stadium Docklands"}
    ),
    Document(
        page_content=(
            "The Gabba (Brisbane Cricket Ground): Located in Woolloongabba, Brisbane, Queensland, with a capacity of 37,000. "
            "It is the fortress of the Brisbane Lions and is known for humid playing conditions and high-scoring matches. "
            "Brisbane recorded a dominant finals run here en route to the 2024 Premiership."
        ),
        metadata={"category": "venues", "topic": "gabba", "title": "The Gabba Brisbane"}
    ),
    Document(
        page_content=(
            "Adelaide Oval: Located in Adelaide, South Australia, with a capacity of 53,500. It hosts the fiercely contested "
            "Showdown derby between the Adelaide Crows and Port Adelaide Power, famous for its historic heritage scoreboard "
            "and loud home crowd advantage."
        ),
        metadata={"category": "venues", "topic": "adelaide_oval", "title": "Adelaide Oval"}
    ),
    Document(
        page_content=(
            "Optus Stadium (Perth Stadium): Located in Burswood, Perth, Western Australia, with a 60,000 capacity. It is the "
            "state-of-the-art home ground for the West Coast Eagles and Fremantle Dockers, hosting the Western Derby and "
            "featuring steep seating bowls and exceptional acoustics."
        ),
        metadata={"category": "venues", "topic": "optus_stadium", "title": "Optus Stadium Perth"}
    ),
    Document(
        page_content=(
            "Sydney Cricket Ground (SCG): Located in Moore Park, Sydney, New South Wales, with a capacity of 48,000. It is the "
            "historic home of the Sydney Swans, characterized by shorter boundaries that emphasize contested footy and rapid inside-50 entries."
        ),
        metadata={"category": "venues", "topic": "scg", "title": "Sydney Cricket Ground (SCG)"}
    ),

    # Clubs & Heritage
    Document(
        page_content=(
            "Collingwood Football Club (The Magpies): Founded in 1892 in Abbotsford/Collingwood, Victoria. Collingwood is "
            "one of the most supported sporting clubs in Australia with 16 VFL/AFL Premierships (including 2023). Home ground: MCG. "
            "Colors: Black and White vertical stripes. Key modern players include Nick Daicos, Scott Pendlebury, Darcy Moore, and Jordan De Goey."
        ),
        metadata={"category": "clubs", "topic": "collingwood", "title": "Collingwood Magpies Club Profile"}
    ),
    Document(
        page_content=(
            "Carlton Football Club (The Blues): Founded in 1864, Carlton is one of the oldest and most successful clubs in "
            "AFL history, holding 16 Premierships. Home grounds: Marvel Stadium and MCG. Colors: Navy Blue with white monogram. "
            "Key leaders include Brownlow Medalist Patrick Cripps, Charlie Curnow, Sam Walsh, and Harry McKay. Major traditional rival: Collingwood."
        ),
        metadata={"category": "clubs", "topic": "carlton", "title": "Carlton Blues Club Profile"}
    ),
    Document(
        page_content=(
            "Brisbane Lions Football Club: Formed in late 1996 through the historic merger of the Fitzroy Lions and the Brisbane Bears. "
            "The Lions achieved a legendary three-peat premiership dynasty (2001, 2002, 2003) and won the 2024 AFL Premiership "
            "under coach Chris Fagan with a resounding 60-point Grand Final triumph over Sydney. Home ground: The Gabba. "
            "Key stars: Lachie Neale (two-time Brownlow Medalist), Hugh McCluggage, Josh Dunkley, and Charlie Cameron."
        ),
        metadata={"category": "clubs", "topic": "brisbane", "title": "Brisbane Lions Club Profile & 2024 Premiership"}
    ),
    Document(
        page_content=(
            "Geelong Football Club (The Cats): Formed in 1859, making it the second-oldest club in Australian rules football. "
            "Known for sustained modern excellence, winning Premierships in 2007, 2009, 2011, and 2022. Home ground: GMHBA Stadium "
            "(Kardinia Park) in Geelong, Victoria. Colors: Navy blue and white hoops. Iconic players: Jeremy Cameron, Patrick Dangerfield, Tom Stewart."
        ),
        metadata={"category": "clubs", "topic": "geelong", "title": "Geelong Cats Club Profile"}
    ),
    Document(
        page_content=(
            "Sydney Swans: Originally founded in 1874 as South Melbourne, the club relocated to Sydney in 1982. The Swans have "
            "claimed 5 Premierships (most recently 2005 and 2012) and reached the 2024 Grand Final after finishing minor premiers. "
            "Home ground: SCG. Star midfielders: Isaac Heeney, Chad Warner, Errol Gulden, and Callum Mills."
        ),
        metadata={"category": "clubs", "topic": "sydney", "title": "Sydney Swans Club Profile"}
    ),
    Document(
        page_content=(
            "Richmond Football Club (The Tigers): Established in 1885, Richmond has won 13 Premierships, including their modern "
            "dynasty triumphs in 2017, 2019, and 2020. Home ground: MCG. Colors: Yellow and Black. Famous for their roar chant and "
            "intense pressure brand of football."
        ),
        metadata={"category": "clubs", "topic": "richmond", "title": "Richmond Tigers Club Profile"}
    ),

    # Awards & Honors
    Document(
        page_content=(
            "The Brownlow Medal (Charles Brownlow Trophy): Awarded annually to the player judged 'Fairest and Best' in the "
            "home-and-away season. Field umpires cast 3, 2, 1 votes after each match. Ineligible players include those suspended by the Tribunal. "
            "Patrick Cripps won the 2024 Brownlow Medal with a record-shattering 45 votes, surpassing Lachie Neale and Nick Daicos."
        ),
        metadata={"category": "awards", "topic": "brownlow", "title": "The Brownlow Medal"}
    ),
    Document(
        page_content=(
            "The Coleman Medal: Presented each season to the AFL player who kicks the most goals during the 23-round "
            "home-and-away season (excluding finals). Recent winners include Charlie Curnow (Carlton, 2022 and 2023) and Jesse Hogan (GWS Giants, 2024 with 69 goals)."
        ),
        metadata={"category": "awards", "topic": "coleman", "title": "The Coleman Medal"}
    ),
    Document(
        page_content=(
            "The Norm Smith Medal: Awarded to the player voted best-on-ground in the AFL Grand Final by an independent panel of experts. "
            "In the 2024 Grand Final, Brisbane Lions midfielder Will Ashcroft claimed the Norm Smith Medal with 30 disposals and a goal."
        ),
        metadata={"category": "awards", "topic": "norm_smith", "title": "The Norm Smith Medal"}
    )
]


class AFLKnowledgeVectorStore(VectorStore):
    """
    Production-ready, deterministic LangChain VectorStore for AFL domain text.
    Provides sub-millisecond TF-IDF cosine-similarity retrieval over unstructured text
    without external GPU or online network download bottlenecks.
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        self._docs: List[Document] = []
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self._matrix = None
        if documents:
            self.add_documents(documents)

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = Document(page_content=text, metadata=meta)
            self._docs.append(doc)
            ids.append(str(len(self._docs) - 1))
        corpus = [d.page_content for d in self._docs]
        self._matrix = self._vectorizer.fit_transform(corpus)
        return ids

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return self.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 3, **kwargs: Any) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k)
        return [doc for doc, _ in results]

    def similarity_search_with_score(self, query: str, k: int = 3, **kwargs: Any) -> List[Tuple[Document, float]]:
        if self._matrix is None or len(self._docs) == 0:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        # Filter top-k matches with non-zero similarity
        ranked_indices = np.argsort(sims)[::-1]
        matches = []
        for idx in ranked_indices:
            score = float(sims[idx])
            if score > 0.02:  # Relevancy threshold
                matches.append((self._docs[idx], score))
            if len(matches) >= k:
                break
        # Fallback if no direct keyword match: return most relevant document
        if not matches and len(self._docs) > 0:
            matches.append((self._docs[ranked_indices[0]], float(sims[ranked_indices[0]])))
        return matches

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Optional[Embeddings] = None, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> "AFLKnowledgeVectorStore":
        vs = cls()
        vs.add_texts(texts, metadatas=metadatas)
        return vs

    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Optional[Embeddings] = None, **kwargs: Any) -> "AFLKnowledgeVectorStore":
        vs = cls()
        vs.add_documents(documents)
        return vs


# Singleton Vector Store Instance
_GLOBAL_VECTOR_STORE: Optional[AFLKnowledgeVectorStore] = None

def get_afl_vector_store() -> AFLKnowledgeVectorStore:
    """Returns or lazily initializes the singleton AFL domain vector store."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        _GLOBAL_VECTOR_STORE = AFLKnowledgeVectorStore.from_documents(AFL_KNOWLEDGE_DOCS)
    return _GLOBAL_VECTOR_STORE
