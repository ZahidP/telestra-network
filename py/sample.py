conn = sqlite3.connect('database.sqlite')
c = conn.cursor()
b = c.execute("SELECT * FROM sqlite_master WHERE type='table';")
conn.commit()
a = b.fetchall()

sample = pd.read_sql_query("""
  SELECT *
  FROM Papers""", conn)
s = " ".join(sample["Title"].values.flatten()).split()

table_names = []
dfs = []
for i in range(0,3):
    table_names.push(a[i][1])
    # pd.read_sql_query("SELECT * from surveys", conn)

papers = pd.read_sql_query("SELECT * from Papers", conn)
authors = pd.read_sql_query("SELECT * from Authors", conn)
paper_authors = pd.read_sql_query("SELECT * from PaperAuthors", conn)

# To Do
# ------
#
# - re-understand MIT probability stuff
# - look up bell curve
# - do some Coursera problems
#   -- time yourself
#   -- maybe do some probability problems from textbook
