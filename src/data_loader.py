"""Data loading module for handling multiple ticket file formats"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from src.config import NEW_TICKETS_PATH, OLD_TICKETS_DIR


class TicketDataLoader:
    """Loads and normalizes ticket data from various formats (CSV, XLSX, JSON)"""

    def __init__(self):
        self.new_tickets_path = NEW_TICKETS_PATH
        self.old_tickets_dir = OLD_TICKETS_DIR

    def load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load tickets from CSV file"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')

    def load_xlsx(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load tickets from XLSX file"""
        df = pd.read_excel(file_path, engine='openpyxl')
        return df.to_dict('records')

    def load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load tickets from JSON file (handles columnar format)"""
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle columnar JSON format (dictionary of arrays)
        if isinstance(data, dict) and all(isinstance(v, (list, dict)) for v in data.values()):
            # Convert columnar format to list of records
            keys = list(data.keys())
            if keys and isinstance(data[keys[0]], dict):
                # Data is indexed by numbers (like "20", "21", etc.)
                indices = sorted(data[keys[0]].keys(), key=lambda x: int(x))
                records = []
                for idx in indices:
                    record = {key: data[key][idx] for key in keys}
                    records.append(record)
                return records
            else:
                # Regular columnar format
                num_records = len(data[keys[0]]) if keys else 0
                return [
                    {key: data[key][i] for key in keys}
                    for i in range(num_records)
                ]

        # Handle list format
        return data if isinstance(data, list) else [data]

    def normalize_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize ticket data to consistent schema"""
        normalized = {
            'ticket_id': str(ticket.get('Ticket ID', ticket.get('ticket_id', 'UNKNOWN'))),
            'issue': str(ticket.get('Issue', ticket.get('issue', ''))),
            'description': str(ticket.get('Description', ticket.get('description', ''))),
            'category': str(ticket.get('Category', ticket.get('category', 'Unknown'))),
            'date': str(ticket.get('Date', ticket.get('date', ''))),
        }

        # Add resolution fields for old tickets
        if 'Resolution' in ticket or 'resolution' in ticket:
            normalized['resolution'] = str(ticket.get('Resolution', ticket.get('resolution', '')))
            normalized['agent_name'] = str(ticket.get('Agent Name', ticket.get('agent_name', '')))

            # Handle various boolean representations for Resolved field
            resolved = ticket.get('Resolved', ticket.get('resolved', False))
            if isinstance(resolved, str):
                normalized['resolved'] = resolved.lower() in ('true', '1', 'yes')
            elif isinstance(resolved, (int, float)):
                normalized['resolved'] = bool(resolved)
            else:
                normalized['resolved'] = bool(resolved)

        return normalized

    def load_all_old_tickets(self, filter_resolved: bool = True) -> List[Dict[str, Any]]:
        """
        Load all old tickets from various format files

        Args:
            filter_resolved: If True, only return tickets where Resolved=True

        Returns:
            List of normalized ticket dictionaries
        """
        all_tickets = []

        # Load from CSV files
        csv_files = list(self.old_tickets_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                tickets = self.load_csv(csv_file)
                all_tickets.extend(tickets)
                print(f"Loaded {len(tickets)} tickets from {csv_file.name}")
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")

        # Load from XLSX files
        xlsx_files = list(self.old_tickets_dir.glob("*.xlsx"))
        for xlsx_file in xlsx_files:
            try:
                tickets = self.load_xlsx(xlsx_file)
                all_tickets.extend(tickets)
                print(f"Loaded {len(tickets)} tickets from {xlsx_file.name}")
            except Exception as e:
                print(f"Warning: Failed to load {xlsx_file.name}: {e}")

        # Load from JSON files
        json_files = list(self.old_tickets_dir.glob("*.json"))
        for json_file in json_files:
            try:
                tickets = self.load_json(json_file)
                all_tickets.extend(tickets)
                print(f"Loaded {len(tickets)} tickets from {json_file.name}")
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")

        # Normalize all tickets
        normalized_tickets = [self.normalize_ticket(t) for t in all_tickets]

        # Filter for resolved tickets if requested
        if filter_resolved:
            normalized_tickets = [
                t for t in normalized_tickets
                if t.get('resolved', False)
            ]
            print(f"\nFiltered to {len(normalized_tickets)} resolved tickets")

        return normalized_tickets

    def load_new_tickets(self) -> List[Dict[str, Any]]:
        """Load new (unresolved) tickets"""
        try:
            tickets = self.load_csv(self.new_tickets_path)
            normalized_tickets = [self.normalize_ticket(t) for t in tickets]
            print(f"Loaded {len(normalized_tickets)} new tickets from {self.new_tickets_path.name}")
            return normalized_tickets
        except Exception as e:
            print(f"Error loading new tickets: {e}")
            return []

    def get_category_statistics(self, tickets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of tickets by category"""
        category_counts = {}
        for ticket in tickets:
            category = ticket.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
