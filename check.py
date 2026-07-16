from firecrawl import Firecrawl

client = Firecrawl(api_key="")
result = client.search("test query", limit=1)
print(result)
