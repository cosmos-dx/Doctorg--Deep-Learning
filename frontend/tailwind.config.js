/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#212121',
        'bg-secondary': '#2f2f2f',
        'bg-tertiary': '#3e3e3e',
        'text-primary': '#ececec',
        'text-secondary': '#b4b4b4',
        'accent': '#10a37f',
        'border': '#4a4a4a',
      },
    },
  },
  plugins: [],
}
