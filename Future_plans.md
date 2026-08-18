for latency the thing to add is that we can fire the RAG call and the first LLM call
and if the LLM tells that it needas the data then we can send that without wasting time 
and we can stream that data to sarvam instead of waiting for user to complete and then sending it
by firing both calls tiogehther i can learn of concurrency as well when to stop the other call when one return a 
response if llm tells not needed tool call then stop that thread
If rag return then wiat for llm call 
if llm returns and rag doest then wait for it and then send
and for that we have to make the system faster and faster
concurreny handling multithreading systems
make it pure streaming like no collection just stream things like server side VAD something like this