import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, ChevronDown, Loader2 } from 'lucide-react';
import LocationAutocomplete from './LocationAutocomplete';

export default function BottomSheet({
    pickup,
    drop,
    onPickupChange,
    onDropChange,
    onSearch,
    loading,
    children
}) {
    const [isExpanded, setIsExpanded] = useState(false);
    const [pickupInput, setPickupInput] = useState('');
    const [dropInput, setDropInput] = useState('');

    useEffect(() => {
        if (pickup?.address) setPickupInput(pickup.address);
    }, [pickup]);

    useEffect(() => {
        if (drop?.address) setDropInput(drop.address);
    }, [drop]);

    const handleSearch = () => {
        if (pickupInput && dropInput) {
            onSearch(pickupInput, dropInput);
            setIsExpanded(true);
        }
    };

    return (
        <>
            {/* Overlay when expanded */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 0.5 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black z-10"
                        onClick={() => setIsExpanded(false)}
                    />
                )}
            </AnimatePresence>

            {/* Bottom Sheet */}
            <motion.div
                className="fixed bottom-0 left-0 right-0 bg-white rounded-t-3xl shadow-2xl z-20"
                initial={{ y: '80%' }}
                animate={{ y: isExpanded ? 0 : '60%' }}
                transition={{ type: 'spring', damping: 30, stiffness: 300 }}
                drag="y"
                dragConstraints={{ top: 0, bottom: 0 }}
                dragElastic={0.2}
                onDragEnd={(e, { offset, velocity }) => {
                    if (offset.y > 100 || velocity.y > 500) {
                        setIsExpanded(false);
                    } else if (offset.y < -100 || velocity.y < -500) {
                        setIsExpanded(true);
                    }
                }}
            >
                {/* Drag Handle */}
                <div className="flex justify-center pt-3 pb-2">
                    <div className="w-12 h-1.5 bg-gray-300 rounded-full" />
                </div>

                <div className="px-5 pb-6 max-h-[85vh] overflow-y-auto">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-xl font-bold text-gray-800">Book Your Ride</h2>
                        {isExpanded && (
                            <button
                                onClick={() => setIsExpanded(false)}
                                className="p-2 hover:bg-gray-100 rounded-full transition"
                            >
                                <ChevronDown size={20} />
                            </button>
                        )}
                    </div>

                    {/* Location Inputs */}
                    <div className="space-y-3 mb-4">
                        {/* Pickup Input */}
                        <LocationAutocomplete
                            type="pickup"
                            value={pickup}
                            onChange={onPickupChange}
                            onSelect={(location) => {
                                onPickupChange(location);
                                setPickupInput(location?.address || '');
                            }}
                            placeholder="Enter pickup (e.g., Indiranagar, MG Road)"
                        />

                        {/* Drop Input */}
                        <LocationAutocomplete
                            type="drop"
                            value={drop}
                            onChange={onDropChange}
                            onSelect={(location) => {
                                onDropChange(location);
                                setDropInput(location?.address || '');
                            }}
                            placeholder="Enter drop (e.g., Koramangala, Whitefield)"
                        />

                        {/* Search Button */}
                        <button
                            onClick={handleSearch}
                            disabled={!pickupInput || !dropInput || loading}
                            className="w-full py-4 bg-ola-primary hover:bg-ola-dark disabled:bg-gray-300 text-white font-bold rounded-xl transition flex items-center justify-center gap-2 shadow-lg"
                        >
                            {loading ? (
                                <>
                                    <Loader2 size={20} className="animate-spin" />
                                    Finding Best Rides...
                                </>
                            ) : (
                                <>
                                    <Search size={20} />
                                    Search Rides
                                </>
                            )}
                        </button>
                    </div>

                    {/* Results Area */}
                    <AnimatePresence>
                        {isExpanded && children && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: 20 }}
                                transition={{ delay: 0.1 }}
                            >
                                {children}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </motion.div>
        </>
    );
}
