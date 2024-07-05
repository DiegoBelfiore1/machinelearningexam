import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Data for POS categories
pos_categories = ['Nouns', 'Verbs', 'Prepositions', 'Possessive pronouns', 'Adjectives']
pos_n = [172723, 66158, 67590, 17221, 23475]
pos_f = [194516, 57123, 32270, 23178, 27776]

# Data for Sentiment, Filler Words, Function Words
sentiment_categories = ['Polarity', 'Function Words', 'Filler Words']
sentiment_n = [172723, 66158, 67590]  # Example data for antisocial behavior
sentiment_f = [194516, 57123, 32270]  # Example data for control group

# Total sample sizes for both analyses
total1 = 540475
total2 = 591513

# Function to perform chi-square test and plot results
def chi_square_test_and_plot(categories, n_values, f_values, title):
    chi2_values = []
    p_values = []

    # Perform chi-square test for each category
    for category, n_val, f_val in zip(categories, n_values, f_values):
        observed = np.array([[n_val, total1 - n_val], [f_val, total2 - f_val]])
        
        # Perform chi-square test
        chi2, p, dof, expected = chi2_contingency(observed)
        
        # Append chi-square statistic and p-value to lists
        chi2_values.append(chi2)
        p_values.append(p)

        # Output results (optional)
        print(f"For category {category}:")
        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p}")
        print()

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(categories)), p_values, marker='o', linestyle='-', color='b', label='P-value')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Significance Level (alpha = 0.05)')
    plt.xlabel('Category')
    plt.ylabel('P-value')
    plt.title(f'P-value vs. Category for {title}')

    # Set x-axis ticks and labels
    plt.xticks(range(len(categories)), categories)

    # Set y-axis limits to focus on the range 0 to 0.1
    plt.ylim(0, 0.1)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Annotate points with their respective values of n and f
    for i, (category, p_val) in enumerate(zip(categories, p_values)):
        plt.annotate(f'n={n_values[i]}, f={f_values[i]}', (i, p_val), textcoords="offset points", xytext=(0,10), ha='center')

    plt.show()

# Perform and plot chi-square test for POS categories
chi_square_test_and_plot(pos_categories, pos_n, pos_f, "POS Categories")

# Perform and plot chi-square test for Sentiment, Filler Words, Function Words
chi_square_test_and_plot(sentiment_categories, sentiment_n, sentiment_f, "Sentiment, Filler Words, Function Words")
