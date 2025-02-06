import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request


app = Flask(__name__)
x = sp.symbols('x')

# Root-Finding Methods
def bisection_method(f, a, b, tol=1e-5):
    iter_count = 0
    guesses = []
    if f.subs(x, a) * f.subs(x, b) > 0:
        return None, iter_count, guesses  # No root between a and b
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        guesses.append(c)
        iter_count += 1
        if f.subs(x, c) == 0:
            return c, iter_count, guesses
        elif f.subs(x, a) * f.subs(x, c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iter_count, guesses


def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    iter_count = 0
    guesses = [x0, x1]
    for _ in range(max_iter):
        f_x0 = f.subs(x, x0)
        f_x1 = f.subs(x, x1)
        if f_x1 - f_x0 == 0:  # Avoid division by zero
            return None, iter_count, guesses
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        guesses.append(x2)
        iter_count += 1
        if abs(x2 - x1) < tol:
            return x2, iter_count, guesses
        x0, x1 = x1, x2
    return None, iter_count, guesses


def false_position_method(f, a, b, tol=1e-5):
    iter_count = 0
    guesses = []
    f_a = f.subs(x, a)
    f_b = f.subs(x, b)
    if f_a * f_b > 0:
        return None, iter_count, guesses  # No root between a and b
    while (b - a) / 2 > tol:
        c = a - f_a * (b - a) / (f_b - f_a)
        guesses.append(c)
        iter_count += 1
        f_c = f.subs(x, c)
        if f_c == 0:
            return c, iter_count, guesses
        elif f_a * f_c < 0:
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c
    return (a + b) / 2, iter_count, guesses


def newton_raphson_method(f, x0, tol=1e-5, max_iter=100):
    iter_count = 0
    guesses = [x0]
    f_prime = sp.diff(f, x)  # Derivative of the function
    for _ in range(max_iter):
        f_x0 = f.subs(x, x0)
        f_prime_x0 = f_prime.subs(x, x0)
        if f_prime_x0 == 0:
            return None, iter_count, guesses  # Derivative is zero, can't continue
        x1 = x0 - f_x0 / f_prime_x0
        guesses.append(x1)
        iter_count += 1
        if abs(x1 - x0) < tol:
            return x1, iter_count, guesses
        x0 = x1
    return None, iter_count, guesses


# Route to handle user input and calculations
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    iterations = 0
    guesses = []
    plot_filename = ""
    if request.method == "POST":
        equation = request.form.get("equation")
        method = request.form.get("method")
        a = float(request.form.get("a"))
        b = float(request.form.get("b"))
        x0 = float(request.form.get("x0"))
        x1 = float(request.form.get("x1"))

        # Convert the equation string to a symbolic expression
        try:
            f = sp.sympify(equation)
        except sp.SympifyError:
            result = "Invalid equation format"
            return render_template("index.html", result=result)

        if method == "bisection":
            result, iterations, guesses = bisection_method(f, a, b)
        elif method == "secant":
            result, iterations, guesses = secant_method(f, x0, x1)
        elif method == "false_position":
            result, iterations, guesses = false_position_method(f, a, b)
        elif method == "newton_raphson":
            result, iterations, guesses = newton_raphson_method(f, x0)
        else:
            result = "Invalid method selected"

        # Create a plot for the method
        if result is not None:
            x_vals = np.linspace(a - 1, b + 1, 400)
            y_vals = np.array([f.subs(x, val) for val in x_vals], dtype=float)

            # Create the plot for visualization
            plt.figure(figsize=(8, 6))
            plt.plot(x_vals, y_vals, label="f(x)")
            plt.axhline(0, color='black', linewidth=0.5)

            # Mark guesses on the plot
            guesses_y = [f.subs(x, g) for g in guesses]
            plt.scatter(guesses, guesses_y, color='red', label="Guesses")

            # Mark the final root
            plt.axvline(result, color='green', linestyle='--', label="Root")

            # Add labels and legends
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f"Root Finding Visualization ({method})")
            plt.legend()

            # Save the plot to static folder
            plot_filename = f"static/plot_{method}.png"
            plt.savefig(plot_filename)
            plt.close()

    return render_template("index.html", result=result, iterations=iterations, plot_filename=plot_filename)


if __name__ == "__main__":
    app.run(debug=True)
