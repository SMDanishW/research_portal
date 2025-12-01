import { User, Bot } from "lucide-react";
import SourceDisplay from "./SourceDisplay";
import { Citation } from "./ChatInterface";
import ReactMarkdown from "react-markdown";

interface MessageBubbleProps {
    message: {
        role: "user" | "assistant";
        content: string;
        citations?: Citation[];
    };
}

export default function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === "user";

    return (
        <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
            <div
                className={`flex max-w-[80%] ${isUser ? "flex-row-reverse" : "flex-row"
                    } gap-3`}
            >
                <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? "bg-blue-600" : "bg-purple-600"
                        }`}
                >
                    {isUser ? (
                        <User className="w-5 h-5 text-white" />
                    ) : (
                        <Bot className="w-5 h-5 text-white" />
                    )}
                </div>

                <div className="flex flex-col gap-2">
                    <div
                        className={`p-4 rounded-2xl ${isUser
                                ? "bg-blue-600 text-white rounded-tr-none"
                                : "bg-gray-800 text-gray-100 rounded-tl-none border border-gray-700"
                            }`}
                    >
                        <div className="prose prose-invert max-w-none text-sm">
                            {/* Simple text rendering for now, could use ReactMarkdown */}
                            <p className="whitespace-pre-wrap">{message.content}</p>
                        </div>
                    </div>

                    {!isUser && message.citations && message.citations.length > 0 && (
                        <div className="bg-gray-800/50 rounded-xl p-3 border border-gray-700/50">
                            <h4 className="text-xs font-semibold text-gray-400 mb-2 uppercase tracking-wider">
                                Sources & Citations
                            </h4>
                            <div className="flex flex-wrap gap-3">
                                {message.citations.map((citation, idx) => (
                                    <SourceDisplay key={idx} citation={citation} />
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
