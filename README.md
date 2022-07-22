### Задача распознавания лиц

Использование предобученной модели InceptionResnetV1 (from https://github.com/timesler/facenet-pytorch). 
Обучение модели на новых данных с использованием CrossEntropyLoss, TripletLoss и ArcFace Loss, сравнение моделей, расчет косинусной близости эмбеддингов.

В папке src представлены функции.

В notebooks -- метрики обученных моделей.
