/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      colors: {
        'gemini-blue': 'rgb(96, 165, 250)',
        'gemini-pink': 'rgb(244, 114, 182)',
      },
    },
  },
  plugins: [],
}