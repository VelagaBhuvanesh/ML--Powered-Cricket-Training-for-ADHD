import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract import Extractor
from transform import Transformer
from load import Loader
from mysql_handler import MySQLHandler
from config_mysql import MYSQL_CONFIG

class ETLPipeline:
    def __init__(self):
        self.run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.extractor = Extractor()
        self.transformer = Transformer()
        self.loader = Loader()
        self.metadata = None
        self.features = None
    
    def run(self):
        """Execute complete ETL pipeline"""
        print("\n" + "🚀"*30)
        print(" STARTING ETL PIPELINE (MySQL)")
        print("🚀"*30)
        print(f"\n  Run ID: {self.run_id}")
        print(f"  Database: {MYSQL_CONFIG['database']} on {MYSQL_CONFIG['host']}")
        print(f"  Raw Data: {os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')}")
        
        try:
            # Step 1: EXTRACT
            print("\n" + "📂"*30)
            print(" STEP 1: EXTRACT")
            print("📂"*30)
            
            if not self.extractor.extract():
                print("\n✗ Pipeline failed during EXTRACT")
                self.loader.close()
                return
            
            self.metadata = self.extractor.get_metadata()
            id_mapping = self.extractor.get_id_mapping()
            
            # Step 2: TRANSFORM
            print("\n" + "⚙️"*30)
            print(" STEP 2: TRANSFORM")
            print("⚙️"*30)
            
            if not self.transformer.transform(self.metadata, id_mapping):
                print("\n✗ Pipeline failed during TRANSFORM")
                self.loader.close()
                return
            
            self.features = self.transformer.get_features()
            
            # Step 3: LOAD
            print("\n" + "💾"*30)
            print(" STEP 3: LOAD")
            print("💾"*30)
            
            summary = self.loader.load(self.metadata, self.features)
            
            # Generate report
            self.generate_report(summary)
            
            print("\n" + "✅"*30)
            print(" ETL PIPELINE COMPLETE!")
            print("✅"*30)
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.loader.close()
    
    def generate_report(self, summary):
        """Generate ETL report"""
        print("\n" + "="*60)
        print("📊 ETL PIPELINE REPORT")
        print("="*60)
        
        extract_stats = self.extractor.get_stats()
        transform_stats = self.transformer.get_stats()
        
        print(f"\n  Run ID: {self.run_id}")
        print(f"  Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n  Statistics:")
        print(f"  - Total participants in metadata: {extract_stats['participants_total']}")
        print(f"  - Participants processed successfully: {transform_stats['participants_processed']}")
        print(f"  - Participants skipped: {transform_stats['participants_skipped']}")
        print(f"  - Windows generated: {transform_stats['windows_generated']:,}")
        
        if transform_stats['skipped_ids']:
            print(f"\n  Skipped Participant IDs: {transform_stats['skipped_ids'][:10]}...")
        
        print("\n  Database Status:")
        print(f"  - Windows in database: {summary['total_windows']:,}")
        print(f"  - Participants in database: {summary['total_participants']}")
        print(f"  - Features stored: {summary['feature_count']}")

def main():
    pipeline = ETLPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()