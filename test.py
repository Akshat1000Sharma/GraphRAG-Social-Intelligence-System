from db.neo4j_client import get_neo4j_client
db = get_neo4j_client()
res = db.run_query("MATCH (u:User {id: 'fb_4224'}) RETURN u.id, u.source_id, u.dataset")
print("fb_4224:", res)

res2 = db.run_query("MATCH (u:User {source_id: '4224'}) RETURN u.id, u.source_id, u.dataset")
print("source_id 4224:", res2)
