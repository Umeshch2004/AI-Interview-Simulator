from env.engine import FALLBACK_QUESTIONS

print("Easy questions in fallback bank:")
for i, q in enumerate(FALLBACK_QUESTIONS['easy']):
    print(f'{i+1}. {q["question"]}')

print("\nMedium questions in fallback bank:")
for i, q in enumerate(FALLBACK_QUESTIONS['medium']):
    print(f'{i+1}. {q["question"]}')

print("\nHard questions in fallback bank:")
for i, q in enumerate(FALLBACK_QUESTIONS['hard']):
    print(f'{i+1}. {q["question"]}')
