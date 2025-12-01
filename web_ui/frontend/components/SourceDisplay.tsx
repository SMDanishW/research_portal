import { FileText, Image as ImageIcon, Table } from "lucide-react";
import { Citation } from "./ChatInterface";

interface SourceDisplayProps {
    citation: Citation;
}

export default function SourceDisplay({ citation }: SourceDisplayProps) {
    const { content_type, content, file_path, page_idx } = citation;

    const getIcon = () => {
        switch (content_type) {
            case "image":
                return <ImageIcon className="w-4 h-4 text-purple-400" />;
            case "table":
                return <Table className="w-4 h-4 text-green-400" />;
            default:
                return <FileText className="w-4 h-4 text-blue-400" />;
        }
    };

    return (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden max-w-xs hover:border-blue-500/50 transition-colors group">
            <div className="bg-gray-800 px-3 py-2 flex items-center justify-between border-b border-gray-700">
                <div className="flex items-center gap-2">
                    {getIcon()}
                    <span className="text-xs font-medium text-gray-300 truncate max-w-[150px]">
                        {file_path.split("/").pop()}
                    </span>
                </div>
                {page_idx !== undefined && (
                    <span className="text-xs text-gray-500 bg-gray-900 px-1.5 py-0.5 rounded">
                        Page {page_idx}
                    </span>
                )}
            </div>

            <div className="p-3">
                {content_type === "image" ? (
                    <div className="relative aspect-video bg-black/20 rounded overflow-hidden">
                        {/* Assuming content is base64 or URL. If it's just description, show text */}
                        {content.startsWith("data:image") || content.startsWith("http") ? (
                            <img
                                src={content}
                                alt="Citation"
                                className="w-full h-full object-contain"
                            />
                        ) : (
                            <p className="text-xs text-gray-400 italic line-clamp-4">{content}</p>
                        )}
                    </div>
                ) : (
                    <p className="text-xs text-gray-400 line-clamp-4 font-mono">
                        {content}
                    </p>
                )}
            </div>
        </div>
    );
}
