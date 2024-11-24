# AI Glossary

A responsive, interactive web application that provides definitions for common artificial intelligence and machine learning terms. Built with React and styled with Tailwind CSS, featuring both light and dark modes for optimal viewing experience.

[Live Demo](https://justinpdaly.github.io/AI-glossary-v1/) <!-- Replace with your actual GitHub Pages URL -->

## Features

- 🔍 Real-time search functionality
- 🌓 Dark/Light mode toggle
- 📱 Responsive design for all device sizes
- 📚 Comprehensive collection of AI/ML terms
- ⚡ Fast, client-side filtering
- 🎯 Accessible UI with ARIA attributes
- 📌 Sticky navigation for better UX

## Technical Stack

- React 17
- Tailwind CSS
- Babel (for JSX transformation)
- Vanilla JavaScript

## Getting Started

### Prerequisites

- A modern web browser
- Basic understanding of HTML and JavaScript

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-glossary.git
cd ai-glossary
```

2. Open `index.html` in your browser:
- You can use a local server like Python's `http.server`:
  ```bash
  python -m http.server 8000
  ```
- Or use any other local development server of your choice

3. The application should now be running at `http://localhost:8000`

### File Structure

```
ai-glossary/
├── index.html
├── app.js
├── data.js
├── README.md
└── screenshot.png
```

## Customization

### Adding New Terms

Add new terms to the `glossaryTerms` array in `data.js`:

```javascript
{
    "term": "Your New Term",
    "definition": "Definition of your new term."
}
```

### Styling

The application uses Tailwind CSS for styling. Modify the classes in `app.js` to customize the appearance.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Terms and definitions compiled from various AI/ML resources
- Built with React and Tailwind CSS
- Hosted on GitHub Pages

---
Made with ❤️ in Melbourne
