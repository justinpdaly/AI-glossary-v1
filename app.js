// Add React hooks usage
const { useState, useEffect } = React;

// Main app component
const AIGlossary = () => {
    const [darkMode, setDarkMode] = useState(false);
    const [selectedTerm, setSelectedTerm] = useState(glossaryTerms[0]);
    const [searchQuery, setSearchQuery] = useState('');

    // Filter terms based on search query
    const filteredTerms = glossaryTerms.filter(item => 
        item.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.definition.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className={`min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
            {/* Header */}
            <header className={`sticky top-0 z-10 p-4 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
                <div className="max-w-6xl mx-auto flex justify-between items-center">
                    <h1 className="text-2xl font-bold">AI Glossary</h1>
                    <button
                        onClick={() => setDarkMode(!darkMode)}
                        className={`p-2 rounded-full ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-100 hover:bg-gray-200'}`}
                        aria-label={darkMode ? "Enable light mode" : "Enable dark mode"}
                    >
                        {darkMode ? "☀️" : "🌙"}
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-6xl mx-auto p-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Sidebar */}
                    <aside className={`md:sticky md:top-24 h-fit max-h-[calc(100vh-6rem)] ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-4`}>
                        <div className="mb-4">
                            <div className="relative">
                                <input
                                    type="search"
                                    placeholder="Search terms..."
                                    className={`w-full px-4 py-2 rounded-md ${
                                        darkMode 
                                            ? 'bg-gray-700 text-white placeholder-gray-400 border-gray-600' 
                                            : 'bg-gray-100 text-gray-900 placeholder-gray-500 border-gray-200'
                                    } border focus:outline-none focus:ring-2 focus:ring-blue-500`}
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                />
                            </div>
                        </div>
                        <nav className="overflow-y-auto h-[calc(100vh-16rem)]">
                            <ul className="space-y-1">
                                {filteredTerms.map((item) => (
                                    <li key={item.term}>
                                        <button
                                            onClick={() => setSelectedTerm(item)}
                                            className={`w-full text-left px-4 py-2 rounded-md transition-colors ${
                                                selectedTerm.term === item.term
                                                    ? (darkMode ? 'bg-blue-600 text-white' : 'bg-blue-100 text-blue-800')
                                                    : (darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100')
                                            }`}
                                            aria-selected={selectedTerm.term === item.term}
                                        >
                                            {item.term}
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        </nav>
                    </aside>

                    {/* Content Area */}
                    <div className={`md:col-span-2 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-md p-6`}>
                        <h2 className="text-2xl font-bold mb-4">
                            {selectedTerm.term}
                        </h2>
                        <p className="text-lg leading-relaxed">
                            {selectedTerm.definition}
                        </p>
                    </div>
                </div>
            </main>
        </div>
    );
};

// Render the app
ReactDOM.render(
    <AIGlossary />,
    document.getElementById('root')
);
