This project was mainly my excuse to finally start using sklearn properly. I wasn't the biggest fan of frameworks at first because when you just instantiate an algorithm as an object and call a couple methods, you don't really see what's going on underneath. The statistics, assumptions, calculations, all the stuff that actually makes the model work gets hidden behind .fit() and .predict(). After spending months grinding through neural networks, maths, optimisation and implementing things manually, it felt weird looking at the abstracted version of everything I'd been studying.

It's kind of backwards honestly. Most people get introduced to machine learning through sklearn, learn the concepts, maybe pick up some of the proofs, and then move into deep learning. I somehow did the opposite. I've always had this habit of forcing myself into the difficult stuff first because once you get through the painful part, everything else tends to make more sense. Surprisingly, that approach actually worked out.

I started with Gaussian Naive Bayes. The breast cancer dataset from sklearn seemed like a decent fit because it's a binary classification problem and most of the features are continuous, which works nicely with the Gaussian assumption. One thing I noticed looking at examples online is that a lot of people just throw everything into a single script. Nothing wrong with that, but I like having things separated out so I made two main classes: BreastCancerData for handling the dataset itself and Training for training, evaluation and plotting.

The actual machine learning part was almost disappointingly simple. After defining the features as attributes and splitting the data into training and testing sets, it was basically a couple method calls and the model was running. I guess that's the whole point of abstraction, but after implementing algorithms manually before, it felt a bit like cheating.

The part I spent the most time on wasn't actually the model, it was figuring out how I wanted to visualise the results. At first I wrote plot_data() just to look at the dataset itself with no predictions involved. Then I found out GaussianNB has a .score() method that gives you the accuracy directly which, again, feels ridiculously abstract. Cool, here's your percentage, goodbye. But I didn't just want a number. I wanted to actually see where the model was succeeding and where it was struggling.

So I wrote plot_both() which plots the prediction results using selected feature relationships. Correct predictions are represented as dots while incorrect predictions are represented as crosses. What I found interesting was that the mistakes weren't randomly scattered all over the graph. They tended to appear in particular regions. To me that suggested the model wasn't just dealing with random noise, it was genuinely struggling to distinguish certain samples.

There are a bunch of possible reasons for that. Maybe radius and texture don't separate those observations very well. Maybe there aren't enough examples in that area of the feature space. Maybe adding more features would help separate them, assuming dimensionality doesn't come along and start causing problems. Hard to say without digging deeper.

Still, it was pretty cool seeing Naive Bayes actually doing something visible instead of just reading about it. Fake Bayes theorem finally came to life. Yes, I know that's not what it's called before anyone gets upset.

After that I went down a rabbit hole with K-Nearest Neighbours and k-fold cross validation.

The thing I liked most about k-fold CV was the idea of giving every part of the dataset a chance to be both training and testing data. Instead of trusting one train-test split, you repeatedly fold the dataset and evaluate across multiple partitions. It just feels like a more honest way of measuring performance because you're not accidentally getting lucky with one split.

For KNN I used the iris dataset. It's small, only 150 samples, which makes it perfect for an algorithm that basically memorises data points and calculates distances whenever you want a prediction. Since the dataset is also pretty clean, I could spend less time preprocessing and more time actually implementing KNN and the cross validation logic.

Plus it just makes intuitive sense. Different flower species should naturally cluster together based on measurements of petals and sepals. It's one of those datasets where you can almost picture the decision boundaries in your head.

I made two plotting methods for this project as well. One visualises how performance changes with different values of k so I could find a reasonable neighbourhood size. The other compares predictions against actual classifications so I could inspect mistakes directly.

The model only made a single error on the test set which resulted in roughly 92% accuracy. For a small dataset and a relatively simple algorithm, that's not terrible at all.

One interesting discussion I had with my boyfriend was about the time complexity of KNN. At first we were throwing around the idea that it might be O(n²), which sounds reasonable when all you think about is distance calculations. But after breaking it down properly, we realised that's not really what's happening.

For every test point, the algorithm calculates distances to all training points across all dimensions, giving a complexity of O(m * n * d), where m is the number of test samples, n is the number of training samples and d is the number of features.

If you're only predicting a single point then m becomes 1, which simplifies things down to O(n * d). Still not amazing for huge datasets, but definitely better than the O(n²) assumption we initially had.

That's kind of the trade-off with KNN. Training is basically free because the model just stores the data, but prediction gets more expensive as the dataset grows. Not exactly something you'd want on a massive dataset, but for smaller datasets or situations where data changes frequently, it's actually pretty practical.

Out of everything in this project, implementing the k-fold cross validation logic was probably the most enjoyable part. Finding the best k value, seeing how performance changed across folds, and then comparing that against the final predictions was a lot more satisfying than I expected. Also the optimal k wasn't absurdly large, which is good because at some point you're basically asking the model to underfit itself into oblivion.

Very cool :)
