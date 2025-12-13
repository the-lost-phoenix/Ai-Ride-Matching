import { useState, useEffect, useRef } from 'react';
import { MapPin, Navigation, Loader2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

export default function LocationAutocomplete({
    type = 'pickup', // 'pickup' or 'drop'
    value,
    onChange,
    onSelect,
    placeholder
}) {
    const [inputValue, setInputValue] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const inputRef = useRef(null);
    const dropdownRef = useRef(null);

    // Update input when value changes from parent
    useEffect(() => {
        if (value?.address) {
            setInputValue(value.address.split(',')[0]); // Show shortened version
        }
    }, [value]);

    // Debounced search for suggestions
    useEffect(() => {
        if (inputValue.length < 2) {
            setSuggestions([]);
            setShowDropdown(false);
            return;
        }

        const timer = setTimeout(() => {
            fetchSuggestions(inputValue);
        }, 300); // 300ms debounce

        return () => clearTimeout(timer);
    }, [inputValue]);

    // Fetch suggestions from Nominatim
    const fetchSuggestions = async (query) => {
        setIsLoading(true);
        try {
            // Add Bangalore context to search
            const searchQuery = `${query}, Bangalore, Karnataka, India`;

            const response = await axios.get(
                `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&countrycodes=IN&limit=10&addressdetails=1`
            );

            if (response.data && response.data.length > 0) {
                // Filter and format results for Bangalore area
                const bangaloreResults = response.data
                    .filter(result => {
                        const lat = parseFloat(result.lat);
                        const lon = parseFloat(result.lon);
                        return lat >= 12.8 && lat <= 13.2 && lon >= 77.4 && lon <= 77.8;
                    })
                    .map(result => ({
                        name: result.name || result.display_name.split(',')[0],
                        address: result.display_name,
                        lat: parseFloat(result.lat),
                        lng: parseFloat(result.lon),
                        type: result.type,
                        class: result.class
                    }));

                setSuggestions(bangaloreResults);
                setShowDropdown(bangaloreResults.length > 0);
            } else {
                setSuggestions([]);
                setShowDropdown(false);
            }
        } catch (error) {
            console.error('Error fetching suggestions:', error);
            setSuggestions([]);
        } finally {
            setIsLoading(false);
        }
    };

    // Handle input change
    const handleInputChange = (e) => {
        const newValue = e.target.value;
        setInputValue(newValue);
        setSelectedIndex(-1);
        onChange?.(null); // Clear selected location when typing
    };

    // Handle suggestion click
    const handleSuggestionClick = (suggestion) => {
        setInputValue(suggestion.name);
        onSelect?.({
            lat: suggestion.lat,
            lng: suggestion.lng,
            address: suggestion.address
        });
        setShowDropdown(false);
        setSuggestions([]);
    };

    // Handle keyboard navigation
    const handleKeyDown = (e) => {
        if (!showDropdown) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev =>
                    prev < suggestions.length - 1 ? prev + 1 : prev
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
                break;
            case 'Enter':
                e.preventDefault();
                if (selectedIndex >= 0 && suggestions[selectedIndex]) {
                    handleSuggestionClick(suggestions[selectedIndex]);
                }
                break;
            case 'Escape':
                setShowDropdown(false);
                break;
            default:
                break;
        }
    };

    // Clear input
    const handleClear = () => {
        setInputValue('');
        onChange?.(null);
        setSuggestions([]);
        setShowDropdown(false);
        inputRef.current?.focus();
    };

    // Click outside to close dropdown
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (
                dropdownRef.current &&
                !dropdownRef.current.contains(e.target) &&
                !inputRef.current?.contains(e.target)
            ) {
                setShowDropdown(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const Icon = type === 'pickup' ? MapPin : Navigation;
    const iconColor = type === 'pickup' ? 'bg-ola-primary' : 'bg-red-500';
    const iconShape = type === 'pickup' ? 'rounded-full' : 'rounded-sm';

    return (
        <div className="relative">
            {/* Input Field */}
            <div className="relative">
                <div className={`absolute left-4 top-1/2 -translate-y-1/2 z-10 w-3 h-3 ${iconColor} ${iconShape} border-2 border-white shadow`} />

                <input
                    ref={inputRef}
                    type="text"
                    placeholder={placeholder}
                    value={inputValue}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    onFocus={() => {
                        if (suggestions.length > 0) setShowDropdown(true);
                    }}
                    className="w-full pl-10 pr-10 py-4 bg-gray-50 rounded-xl border-2 border-transparent focus:border-ola-primary focus:bg-white transition outline-none text-sm font-medium"
                    autoComplete="off"
                />

                {/* Loading or Clear Button */}
                <div className="absolute right-4 top-1/2 -translate-y-1/2">
                    {isLoading ? (
                        <Loader2 size={18} className="animate-spin text-ola-primary" />
                    ) : inputValue && (
                        <button
                            onClick={handleClear}
                            className="text-gray-400 hover:text-gray-600 transition"
                        >
                            <X size={18} />
                        </button>
                    )}
                </div>
            </div>

            {/* Dropdown Suggestions */}
            <AnimatePresence>
                {showDropdown && suggestions.length > 0 && (
                    <motion.div
                        ref={dropdownRef}
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.2 }}
                        className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-2xl border border-gray-200 max-h-80 overflow-y-auto z-50"
                    >
                        {suggestions.map((suggestion, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: index * 0.03 }}
                                onClick={() => handleSuggestionClick(suggestion)}
                                className={`
                  flex items-start gap-3 px-4 py-3 cursor-pointer transition
                  ${index === selectedIndex ? 'bg-ola-primary/10' : 'hover:bg-gray-50'}
                  ${index !== suggestions.length - 1 ? 'border-b border-gray-100' : ''}
                `}
                            >
                                <div className={`mt-1 w-8 h-8 ${iconColor} ${iconShape} flex items-center justify-center flex-shrink-0`}>
                                    <Icon size={16} className="text-white" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <p className="font-semibold text-gray-900 truncate">
                                        {suggestion.name}
                                    </p>
                                    <p className="text-xs text-gray-500 truncate">
                                        {suggestion.address}
                                    </p>
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>

            {/* No results message */}
            <AnimatePresence>
                {showDropdown && !isLoading && suggestions.length === 0 && inputValue.length >= 2 && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-lg border border-gray-200 p-4 z-50"
                    >
                        <p className="text-sm text-gray-500 text-center">
                            No locations found in Bangalore. Try a different search.
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
