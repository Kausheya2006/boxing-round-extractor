### How I solved the longer than 3 minutes round

- Earlier when i used 50 samples, i got only 3 samples per round : three samples are not enough to decide which of them is faulty in case of a ocr-failure

- so we used logic of using 150 samples : so i get 9-10 samples per round : so even if one is faulty rest of them falls under clusters of size > 2
    - cluster0 : [a]
    - cluster1 : [b,c,d]  # say it predicts start-time t1, end-time t'1
    - cluster2 : [e,f]  # say it predicts start-time t2, end-time t'2
    
so start-time : min(t1, t2), end-time : max(t'1, t'2)


## Still issues : 

### Fight4 round 4

actual start-time : 16:04   predicted start-time : 16:11

Reason : the round got paused at earlier phase of the round, so only one sample captured it.

I got : 

```
cluster0 : [1145.9] (Size: 1)   -> 16:04
cluster1 : [1151.1, 1151.3, 1151.4, 1151.5, 1151.6, 1151.7, 1151.8, 1151.9, 1152.0] (Size: 9)  -> 16:11
```

But since cluster0 is earlier, but being size-0 get it rejected, so predicted start-time : 16:11

But if I remove this "size=1 reject" logic, I miss the point of removing OCR-hallucination (assuming there is atmost one hallucination)

#### Suggestive fix : use 250 samples  -> more time-consuming


### Fight4 round 5 

actual end-time : 19:32  predicted end-time : 19:14

Pause happened close to end, so no sample was captured.

### Other issue : full round in case of KO is captured (3 minutes)

#### Why not just use audio for unverified timestamps?

there are cases where clap model creates array [F, F, ..., F] even when there was a genuine ring bell, so we cant just use clap as single mode of verification.

What if we stretch over till start of next round? : we wont know the actual start time of next round r+1 until it is not verified by audio.
So when we are at round r, we have not verified r+1, so we cant rely on next round start time.