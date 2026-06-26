import asyncio

class CitationService:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.paper_to_id = {}
        self.counter = 0

    async def batch_get_ids(self, paper_ids):
        async with self.lock:
            result = {}
            for pid in paper_ids:
                if pid not in self.paper_to_id:
                    self.counter += 1
                    self.paper_to_id[pid] = self.counter
                result[pid] = self.paper_to_id[pid]
            return result


# global registry (per experiment)
citation_services = {}

def get_citation_service(experiment_id: str) -> CitationService:
    if experiment_id not in citation_services:
        citation_services[experiment_id] = CitationService()
    return citation_services[experiment_id]