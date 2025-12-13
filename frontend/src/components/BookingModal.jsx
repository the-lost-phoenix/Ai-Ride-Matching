import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, X, MapPin, Navigation, Clock, DollarSign, Car } from 'lucide-react';
import { useEffect, useState } from 'react';

export default function BookingModal({ isOpen, onClose, vehicle, pickup, drop, distance }) {
    const [stage, setStage] = useState('searching'); // searching -> found -> confirmed

    useEffect(() => {
        if (isOpen) {
            setStage('searching');

            // Simulate finding driver
            const timer1 = setTimeout(() => {
                setStage('found');
            }, 2000);

            // Auto confirm after finding
            const timer2 = setTimeout(() => {
                setStage('confirmed');
            }, 3500);

            return () => {
                clearTimeout(timer1);
                clearTimeout(timer2);
            };
        }
    }, [isOpen]);

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
                        onClick={onClose}
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 50 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 50 }}
                        className="fixed inset-x-4 top-1/2 -translate-y-1/2 max-w-md mx-auto bg-white rounded-3xl shadow-2xl z-50 overflow-hidden"
                    >
                        {/* Close Button */}
                        <button
                            onClick={onClose}
                            className="absolute top-4 right-4 p-2 hover:bg-gray-100 rounded-full transition z-10"
                        >
                            <X size={20} />
                        </button>

                        {/* Content based on stage */}
                        <div className="p-8">
                            {stage === 'searching' && (
                                <div className="text-center">
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                                        className="mx-auto w-20 h-20 mb-6"
                                    >
                                        <div className="w-full h-full rounded-full border-4 border-gray-200 border-t-ola-primary"></div>
                                    </motion.div>
                                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Finding Your Driver...</h2>
                                    <p className="text-gray-600">Looking for nearby {vehicle?.type} vehicles</p>

                                    <motion.div
                                        className="mt-6 flex justify-center gap-2"
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                    >
                                        {[1, 2, 3].map((i) => (
                                            <motion.div
                                                key={i}
                                                className="w-3 h-3 bg-ola-primary rounded-full"
                                                animate={{ y: [0, -10, 0] }}
                                                transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                                            />
                                        ))}
                                    </motion.div>
                                </div>
                            )}

                            {stage === 'found' && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="text-center"
                                >
                                    <div className="mx-auto w-20 h-20 bg-ola-primary/10 rounded-full flex items-center justify-center mb-6">
                                        <Car size={40} className="text-ola-primary" />
                                    </div>
                                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Driver Found!</h2>
                                    <p className="text-gray-600 mb-6">Confirming your booking...</p>

                                    <div className="bg-gray-50 rounded-2xl p-4 text-left space-y-3">
                                        <div className="flex items-center gap-3">
                                            <div className="w-12 h-12 bg-gray-300 rounded-full flex items-center justify-center text-xl">
                                                👨‍✈️
                                            </div>
                                            <div>
                                                <p className="font-semibold text-gray-800">Rajesh Kumar</p>
                                                <p className="text-sm text-gray-600">⭐ 4.8 • KA-01-AB-1234</p>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}

                            {stage === 'confirmed' && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="text-center"
                                >
                                    <motion.div
                                        initial={{ scale: 0 }}
                                        animate={{ scale: 1 }}
                                        transition={{ type: 'spring', stiffness: 200, damping: 15 }}
                                        className="mx-auto w-20 h-20 bg-ola-primary rounded-full flex items-center justify-center mb-6"
                                    >
                                        <CheckCircle size={50} className="text-white" />
                                    </motion.div>

                                    <h2 className="text-2xl font-bold text-gray-800 mb-2">Booking Confirmed! 🎉</h2>
                                    <p className="text-gray-600 mb-6">Your driver is on the way</p>

                                    {/* Booking Details */}
                                    <div className="bg-gradient-to-br from-ola-primary/10 to-green-50 rounded-2xl p-6 text-left space-y-4 mb-6">
                                        <div className="flex items-start gap-3">
                                            <MapPin size={20} className="text-ola-primary mt-0.5 flex-shrink-0" />
                                            <div className="flex-1">
                                                <p className="text-xs text-gray-600 mb-0.5">Pickup</p>
                                                <p className="font-medium text-sm text-gray-800">{pickup?.address || 'Your location'}</p>
                                            </div>
                                        </div>

                                        <div className="border-l-2 border-dashed border-gray-300 h-6 ml-2"></div>

                                        <div className="flex items-start gap-3">
                                            <Navigation size={20} className="text-red-500 mt-0.5 flex-shrink-0" />
                                            <div className="flex-1">
                                                <p className="text-xs text-gray-600 mb-0.5">Drop</p>
                                                <p className="font-medium text-sm text-gray-800">{drop?.address || 'Destination'}</p>
                                            </div>
                                        </div>

                                        <div className="border-t pt-4 mt-4 flex justify-between items-center">
                                            <div className="flex items-center gap-2 text-sm text-gray-700">
                                                <Clock size={16} className="text-ola-primary" />
                                                <span>{vehicle?.eta} mins</span>
                                            </div>
                                            <div className="flex items-center gap-2 text-sm text-gray-700">
                                                <Car size={16} className="text-ola-primary" />
                                                <span>{vehicle?.type}</span>
                                            </div>
                                            <div className="text-lg font-bold text-gray-900">
                                                ₹{vehicle?.price}
                                            </div>
                                        </div>
                                    </div>

                                    <motion.button
                                        whileHover={{ scale: 1.05 }}
                                        whileTap={{ scale: 0.95 }}
                                        onClick={onClose}
                                        className="w-full py-4 bg-ola-primary hover:bg-ola-dark text-white font-bold rounded-xl transition"
                                    >
                                        Track Your Ride
                                    </motion.button>
                                </motion.div>
                            )}
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
