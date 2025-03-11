const {Pinecone} = require('@pinecone-database/pinecone')
const {OpenAI} = require('openai')
const dotenv = require('dotenv')
dotenv.config({})

const pineconeClient = new Pinecone({
    apiKey:process.env.PINE_CONE_API_KEY
})

const openAiClient = new OpenAI({
    apiKey:process.env.OpenAI_API_KEY
})

const studentInfo = `Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
 is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
 in her free time in hopes of working at a tech company after graduating from the University of Washington.`;
 
 const clubInfo = `The university chess club provides an outlet for students to come together and enjoy playing
 the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
 the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
 participate in tournaments, analyze famous chess matches, and improve members' skills.`;
 
 const universityInfo = `The University of Washington, founded in 1861 in Seattle, is a public research university
 with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
 As the flagship institution of the six public universities in Washington state,
 UW encompasses over 500 buildings and 20 million square feet of space,
 including one of the largest library systems in the world.`;


 const vectorDataInp = [
        {
            info: studentInfo,
            reference: "some student 123",
            relevance: 0.9
        },
        {
            info: clubInfo,
            reference: "some club 456",
            relevance: 0.8
        },
        {
            info: universityInfo,
            reference: "some university 789",
            relevance: 0.7
        }, 
 ]

 //create new pinecone index

 async function createPineconeIndex(indexName){
    const index = await pineconeClient.createIndex({
        name:indexName,
        dimension:1536,
        metric:"cosine",
        spec:{
            serverless:{
                cloud:"aws",
                region:"us-east-1"
            }
        }
    })
    return index;
 }

 async function insertDataIntoPineconeIndex(index,data){
    try{
        await index.upsert([{
            id:data.id,
            values:data.embedData,
            metadata:data.metadata
        }])
        console.log("insertion successs")
    }
    catch(err){
        console.log("Insertion error")
        console.log(err)
    }
 }


 async function upsertDataIntoPineconeIndex(indexName){
    try{
        const index = pineconeClient.index(indexName)
        await Promise.all(vectorDataInp.map(async (data) => {
            console.log(data)
            //first create embedding
            const embedResult = await openAiClient.embeddings.create({
                model:"text-embedding-ada-002",
                input:data.info
            })
            const embedData = embedResult.data[0].embedding
            //insert embedding into pinecone
            await insertDataIntoPineconeIndex(index,{
                id:`${Math.random() * 1000}`,
                embedData,
                metadata:data
            })
        }))
    }
    catch(err){
        console.log("Upsertion error")
        console.log(err)
    }
 }

 async function queryEmbedding(indexName,input){
    //get index;
    const index = pineconeClient.index(indexName)
    //create an embedding of input
    const embedResult = await openAiClient.embeddings.create({
        model:"text-embedding-ada-002",
        input
    })
    const embedData = embedResult.data[0].embedding
    //query into pinecone for similarity match
    const queryResult = await index.query({
        vector:embedData,
        topK:1,
        includeMetadata:true,
        includeValues:true
    })

    return queryResult && queryResult.matches[0]
 }

 async function questionAndAnswer(question){
    const embeddingQueryRes = await queryEmbedding('learn-pinecone',question)
    const metadata = embeddingQueryRes && embeddingQueryRes.metadata;
    let answer = null;
    console.log(metadata)
     //now ask open with context injection
    if(metadata){
        const response = await openAiClient.chat.completions.create({
            model:'gpt-3.5-turbo',
            temperature:0,
            messages:[
                {
                    role:'assistant',
                    content:`Answer the next question using the following information:\n${JSON.stringify(metadata, null, 2)}`
                },{
                    role:'user',
                    content:question
                }
            ]
        })
        const responseMessage = response && response.choices && response.choices[0] && response.choices[0].message;
        answer = responseMessage
    }
    return answer

 }

async function main(){
    //create a pinecone index
    // const index = await createPineconeIndex('learn-pinecone')
    //upsert my data into pinecone
    // const data = await upsertDataIntoPineconeIndex('learn-pinecone')
   const ans = await questionAndAnswer("What does Alexandra Thompson like to do in her free time?")
   console.log(ans)
}

main()