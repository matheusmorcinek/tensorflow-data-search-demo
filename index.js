const use = require('@tensorflow-models/universal-sentence-encoder');
const tf = require('@tensorflow/tfjs');
const functionalities = require('./items.json');

const { performance } = require('perf_hooks');

function cosineSimilarity(A, B) {
    const dotProduct = tf.sum(tf.mul(A, B));
    const normA = tf.sqrt(tf.sum(tf.square(A)));
    const normB = tf.sqrt(tf.sum(tf.square(B)));
    return dotProduct.div(normA.mul(normB));
};

async function findSimilarFunctionalities(query, options) {

    let totalStartTime = performance.now();

    let initialMemoryUsage = process.memoryUsage();
    console.log(`Initial Memory Usage: RSS=${initialMemoryUsage.rss} bytes, Heap Total=${initialMemoryUsage.heapTotal} bytes, Heap Used=${initialMemoryUsage.heapUsed} bytes`);

    const { keys, numResults, threshold } = options;

     // Load the Universal Sentence Encoder model
     let startTime = performance.now();
     const model = await use.load();
     let endTime = performance.now();
     console.log(`Time taken to load the model: ${endTime - startTime} ms (${((endTime - startTime) / 1000).toFixed(2)} seconds)`);
 

    // Combine specified keys for embeddings
    const combinedTexts = functionalities.map(func => {
        return keys.map(key => {
            const value = func[key];
            return Array.isArray(value) ? value.join(' ').toLowerCase().trim() : value.toLowerCase().trim();
        }).join(' ');
    });

    //console.log(combinedTexts)

    const combinedQuery = query.toLowerCase().trim();

    startTime = performance.now();
    const descriptionEmbeddings = await model.embed(combinedTexts);
    endTime = performance.now();
    console.log(`Time taken for description embeddings: ${endTime - startTime} ms (${((endTime - startTime) / 1000).toFixed(2)} seconds)`);

    // Generate embedding for the query
    startTime = performance.now();
    const queryEmbedding = await model.embed([combinedQuery]);
    endTime = performance.now();
    console.log(`Time taken for query embedding: ${endTime - startTime} ms (${((endTime - startTime) / 1000).toFixed(2)} seconds)`);


    // Calculate cosine similarity between query and combined texts
    const scores = [];
    for (let i = 0; i < combinedTexts.length; i++) {
        const score = cosineSimilarity(queryEmbedding, descriptionEmbeddings.gather([i])).dataSync()[0];
        scores.push({ index: i, score: score });
    };

    // Sort scores in descending order (higher cosine similarity means higher relevance)
    scores.sort((a, b) => b.score - a.score);

    // Output results based on the highest score and threshold
    console.log(`Query: ${query}\n`);
    if (scores[0].score >= threshold) {
        console.log("Most relevant functionality with high confidence:");
        const idx = scores[0].index;
        console.log(`Functionality ID: ${functionalities[idx].functionality_id}, Name: ${functionalities[idx].name}, Score: ${scores[0].score.toFixed(4)}`);
    } else {
        if (numResults === 1) {
            console.log("Most similar functionality:");
        } else {
            console.log(`Top ${numResults} similar functionalities:`);
        }
        for (let i = 0; i < numResults && i < scores.length; i++) {
            const idx = scores[i].index;
            console.log(`Functionality ID: ${functionalities[idx].functionality_id}, Name: ${functionalities[idx].name}, Score: ${scores[i].score.toFixed(4)}`);
        }
    };

    let totalEndTime = performance.now();
    console.log(`Total execution time: ${totalEndTime - totalStartTime} ms (${((totalEndTime - totalStartTime) / 1000).toFixed(2)} seconds)`);
};

const options = {
    keys: [
        "name",
        "description",
        "searchTokens",
        "category"
    ],
    numResults: 3, // number of top results
    threshold: 0.5 // confidence threshold
};

//organize my discount coupons
const query = "organize my coupons";
findSimilarFunctionalities(query, options);
