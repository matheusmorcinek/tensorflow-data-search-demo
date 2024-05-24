npm install 
node index.js

edit the function findSimilarFunctionalities with different QUERIES and OPTIONS.

the findSimilarFunctionalities function includes OPTIONS to customize the search behavior:
keys: Specifies which attributes of the functionalities to include in the embedding (e.g., name, description, searchTokens, category).
numResults: Defines the number of top results to return.
threshold: Sets a confidence threshold for the similarity score to determine if a functionality is considered a high-confidence match. If a match exceeds this threshold, the function returns the single most relevant functionality. Otherwise, it shows the top numResults that the we define to find.
